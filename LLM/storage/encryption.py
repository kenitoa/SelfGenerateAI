"""Private local key provider and bounded, authenticated backup file streams.

Keys are deliberately absent from records, manifests, events and backup bundles.
The key directory can be injected by the operator; it is never a model input.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from LLM.core import Fault, canonical, file_hash, require

MAGIC = b'SEED-AESGCM-CHUNK1\n'
CHUNK = 1024 * 1024
KEY_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


def _windows_permissions(path, apply=False):
    # No shell-built command and no inherited Users/Everyone permissions.
    script = r'''$ErrorActionPreference='Stop';$p=$env:SEED_PERMISSION_PATH;
$u=[Security.Principal.WindowsIdentity]::GetCurrent();$ids=@($u.User.Value,'S-1-5-18','S-1-5-32-544');
if($env:SEED_PERMISSION_APPLY -eq '1'){
 $a=if(Test-Path -LiteralPath $p -PathType Container){[IO.Directory]::GetAccessControl($p,[Security.AccessControl.AccessControlSections]::Access)}else{[IO.File]::GetAccessControl($p,[Security.AccessControl.AccessControlSections]::Access)};
 $a.SetAccessRuleProtection($true,$false);
 foreach($rule in @($a.Access)){$a.RemoveAccessRuleSpecific($rule)};
 foreach($id in $ids){$sid=New-Object Security.Principal.SecurityIdentifier($id);$r=New-Object Security.AccessControl.FileSystemAccessRule($sid,'FullControl','Allow');$a.AddAccessRule($r)};
 if(Test-Path -LiteralPath $p -PathType Container){[IO.Directory]::SetAccessControl($p,$a)}else{[IO.File]::SetAccessControl($p,$a)}
};$a=Get-Acl -LiteralPath $p;$bad=@($a.Access|Where-Object{$_.AccessControlType -eq 'Allow' -and $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -notin $ids});
@{protected=$a.AreAccessRulesProtected;safe=($bad.Count -eq 0);owner=($a.Owner -eq $u.Name -or $a.Owner -eq $u.User.Value)}|ConvertTo-Json -Compress'''
    env = {**os.environ, 'SEED_PERMISSION_PATH': str(path), 'SEED_PERMISSION_APPLY': '1' if apply else '0'}
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                            env=env, capture_output=True, timeout=20, text=True)
    try:
        receipt = json.loads(result.stdout) if result.returncode == 0 else {}
    except ValueError:
        receipt = {}
    require(receipt.get('protected') and receipt.get('safe') and receipt.get('owner'), 'BACKUP_KEY_PERMISSIONS_UNSAFE', status=403)


def private(path, *, directory=False, create=False):
    path = Path(path)
    require(not path.is_symlink(), 'BACKUP_KEY_PATH_UNSAFE', status=403)
    if create and directory:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(path.exists() and all(not item.is_symlink() for item in (path, *path.parents)), 'BACKUP_KEY_PATH_UNSAFE', status=403)
    if os.name == 'nt':
        _windows_permissions(path, apply=create)
    else:
        if create:
            path.chmod(0o700 if directory else 0o600)
        info = path.stat()
        require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0,
                'BACKUP_KEY_PERMISSIONS_UNSAFE', status=403)
    return path


def key_directory(store):
    injected = os.environ.get('SEED_BACKUP_KEY_DIRECTORY')
    root = Path(injected).absolute() if injected else Path(store.root)/'metadata'/'privatekeys'
    require(not root.resolve().is_relative_to((Path(store.root)/'backups').resolve()), 'BACKUP_KEY_PATH_UNSAFE')
    # Injected directories belong to an operator/provider and must already be private.
    return private(root, directory=True, create=not injected)


def get_key(store, key_id, *, create=False):
    require(isinstance(key_id, str) and KEY_ID.fullmatch(key_id), 'BACKUP_KEY_ID_INVALID')
    root = key_directory(store)
    path = root/(key_id+'.key')
    if create:
        require(not path.exists(), 'BACKUP_KEY_ALREADY_EXISTS', status=409)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        private(path, create=True)
        with path.open('wb') as stream:
            stream.write(os.urandom(32)); stream.flush(); os.fsync(stream.fileno())
    else:
        require(path.exists(), 'BACKUP_KEY_UNAVAILABLE', status=409)
        private(path)
    key = path.read_bytes()
    require(len(key) == 32, 'BACKUP_KEY_INVALID')
    return key


def destroy_key(store, key_id):
    require(isinstance(key_id, str) and KEY_ID.fullmatch(key_id), 'BACKUP_KEY_ID_INVALID')
    path = key_directory(store)/(key_id+'.key')
    if path.exists():
        private(path)
        path.unlink()


def encrypt_file(source, target, key, binding):
    aes = AESGCM(key)
    prefix = os.urandom(8)
    plain = hashlib.sha256()
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:
        dst.write(MAGIC+prefix)
        index = 0
        while True:
            block = src.read(CHUNK)
            # An authenticated empty final block binds truncation and exact length.
            plain.update(block)
            aad = canonical([binding, index, not block]).encode()
            encrypted = aes.encrypt(prefix+index.to_bytes(4, 'big'), block, aad)
            dst.write(struct.pack('>I', len(encrypted))); dst.write(encrypted)
            index += 1
            require(index < 2**32, 'BACKUP_FILE_TOO_LARGE')
            if not block:
                break
        dst.flush(); os.fsync(dst.fileno())
    return {'plaintextHash': plain.hexdigest(), 'ciphertextHash': file_hash(target)}


def decrypt_file(source, target, key, binding):
    aes = AESGCM(key)
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(source).open('rb') as src, target.open('xb') as dst:
            require(src.read(len(MAGIC)) == MAGIC, 'BACKUP_ENCRYPTION_INVALID')
            prefix = src.read(8); require(len(prefix) == 8, 'BACKUP_ENCRYPTION_INVALID')
            index = 0
            while True:
                length_bytes = src.read(4)
                require(len(length_bytes) == 4, 'BACKUP_ENCRYPTION_INVALID')
                length = struct.unpack('>I', length_bytes)[0]
                require(16 <= length <= CHUNK+16, 'BACKUP_ENCRYPTION_INVALID')
                block = src.read(length); require(len(block) == length, 'BACKUP_ENCRYPTION_INVALID')
                final = length == 16
                try:
                    plain = aes.decrypt(prefix+index.to_bytes(4, 'big'), block, canonical([binding, index, final]).encode())
                except InvalidTag:
                    raise Fault('BACKUP_AUTHENTICATION_FAILED', status=409) from None
                dst.write(plain); index += 1
                require(index < 2**32, 'BACKUP_FILE_TOO_LARGE')
                if final:
                    require(not src.read(1), 'BACKUP_ENCRYPTION_INVALID')
                    break
            dst.flush(); os.fsync(dst.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return file_hash(target)
