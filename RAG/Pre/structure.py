"""Static syntax trees and bounded code relationships; source is never executed."""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import PurePosixPath
import re

GRAMMARS={'python':('python','language'),'javascript':('javascript','language'),'typescript':('typescript','language_typescript'),
          'tsx':('typescript','language_tsx'),'go':('go','language'),'rust':('rust','language'),'java':('java','language'),
          'c':('c','language'),'cpp':('cpp','language'),'csharp':('c_sharp','language')}
DEFINITIONS={'function_definition','async_function_definition','class_definition','function_declaration','class_declaration',
             'method_definition','method_declaration','function_item','struct_item','enum_item','trait_item','interface_declaration',
             'type_alias_declaration','type_declaration','struct_specifier','class_specifier','namespace_definition','constructor_declaration'}
IMPORTS={'import_statement','import_from_statement','import_declaration','use_declaration','preproc_include','using_directive'}
CALLS={'call','call_expression','method_invocation','object_creation_expression'}

@lru_cache(maxsize=16)
def grammar(language):
    from tree_sitter import Language
    module,function=GRAMMARS[language]
    return Language(getattr(importlib.import_module('tree_sitter_'+module),function)())

def _name(node,raw):
    field=node.child_by_field_name('name') or node.child_by_field_name('declarator')
    while field and field.type in ('function_declarator','pointer_declarator','reference_declarator'):
        field=field.child_by_field_name('declarator')
    return raw[field.start_byte:field.end_byte].decode('utf-8')[:160] if field else ''

def _parse(text,language,line_offset=0,char_offset=0):
    if language not in GRAMMARS:return {'parser':'unsupported','symbols':[],'imports':[],'calls':[],'syntaxValid':None}
    from tree_sitter import Parser
    raw=text.encode();tree=Parser(grammar(language)).parse(raw)
    symbols=[];imports=[];calls=[];stack=[(tree.root_node,None)];visited=0
    while stack and visited<100000:
        node,owner=stack.pop();visited+=1
        if node.type in DEFINITIONS or (node.type=='variable_declarator' and any(c.type in ('arrow_function','function_expression') for c in node.named_children)):
            name=_name(node,raw)
            if name and len(symbols)<1000:
                qualified=(owner+'.' if owner else '')+name
                # Byte offsets avoid allocating native Point objects during traversal.
                item={'name':name,'qualifiedName':qualified,'kind':node.type,'startLine':raw.count(b'\n',0,node.start_byte)+1+line_offset,
                      'endLine':raw.count(b'\n',0,node.end_byte)+1+line_offset,'startOffset':char_offset+len(raw[:node.start_byte].decode()),
                      'endOffset':char_offset+len(raw[:node.end_byte].decode()),'language':language,'parent':owner}
                symbols.append(item);owner=qualified
        if node.type in IMPORTS and len(imports)<200:
            imports.append({'text':raw[node.start_byte:node.end_byte].decode()[:1000],'line':raw.count(b'\n',0,node.start_byte)+1+line_offset})
        if node.type in CALLS and len(calls)<2000:
            field=node.child_by_field_name('function') or node.child_by_field_name('name') or node.child_by_field_name('type')
            if field:
                target=raw[field.start_byte:field.end_byte].decode()[:160]
                calls.append({'target':target,'name':re.split(r'[.:>]+',target)[-1],'caller':owner,'line':raw.count(b'\n',0,node.start_byte)+1+line_offset})
        stack.extend((child,owner) for child in reversed(node.named_children))
    return {'parser':'tree-sitter','symbols':symbols,'imports':imports,'calls':calls,'syntaxValid':not tree.root_node.has_error,'truncated':bool(stack)}

def analyze(text,language):
    if language not in ('markdown','html'):
        return _parse(text,language)
    result={'parser':'fenced-tree-sitter','symbols':[],'imports':[],'calls':[],'syntaxValid':None}
    validity=[]
    pattern=re.compile(r'(?m)^(`{3,}|~{3,})([\w+#-]*)[^\n]*\n(.*?)^\1[ \t]*$',re.S)
    aliases={'py':'python','js':'javascript','ts':'typescript','cs':'csharp','c++':'cpp','rs':'rust'}
    for block in list(pattern.finditer(text))[:80]:
        lang=aliases.get(block[2].lower(),block[2].lower());code=block[3]
        if not lang:
            lang='python' if re.search(r'(?m)^\s*(?:async\s+)?def |^from .+ import ',code) else 'javascript' if 'function ' in code else 'unknown'
        part=_parse(code,lang,text.count('\n',0,block.start(3)),block.start(3))
        for key in ('symbols','imports','calls'):result[key].extend(part[key])
        if part['syntaxValid'] is not None:validity.append(part['syntaxValid'])
    result['syntaxValid']=all(validity) if validity else None
    return result

def chunk_structure(part,analysis):
    symbols=[s for s in analysis['symbols'] if s['startLine']<=part['endLine'] and s['endLine']>=part['startLine']]
    owner=min(symbols,key=lambda s:s['endLine']-s['startLine']) if symbols else None
    return {'symbol':owner['name'] if owner else part.get('symbol',''),'qualifiedSymbol':owner['qualifiedName'] if owner else '',
            'defines':list(dict.fromkeys(s['name'] for s in symbols)),
            'calls':list(dict.fromkeys(c['name'] for c in analysis['calls'] if part['startLine']<=c['line']<=part['endLine'])),
            'imports':analysis['imports'][:200],'structureParser':analysis['parser']}

def related(anchors,permitted,docs,sources,limit=8):
    def group(chunk):
        doc=docs[chunk['documentId']];source=sources[doc['sourceId']]
        return source.get('repository') or source.get('repositoryFamily') or doc.get('projectId') or source['id']
    by_symbol={};by_module={};by_call={}
    for chunk in permitted.values():
        for name in chunk.get('defines',[]):by_symbol.setdefault((group(chunk),name),[]).append(chunk)
        by_module.setdefault((group(chunk),PurePosixPath(docs[chunk['documentId']]['path']).stem),[]).append(chunk)
        for name in chunk.get('calls',[]):by_call.setdefault((group(chunk),name),[]).append(chunk)
    edges=[];seen=set()
    for identifier in anchors[:8]:
        anchor=permitted[identifier]
        for name in anchor.get('calls',[])[:32]:
            for target in by_symbol.get((group(anchor),name),[])[:4]:
                if target['id']==identifier or target['id'] in seen:continue
                seen.add(target['id']);edges.append({'from':identifier,'to':target['id'],'kind':'calls','symbol':name})
                if len(edges)>=limit:return edges
        for name in anchor.get('defines',[])[:32]:
            for target in by_call.get((group(anchor),name),[])[:4]:
                if target['id']==identifier or target['id'] in seen:continue
                seen.add(target['id']);edges.append({'from':identifier,'to':target['id'],'kind':'called_by','symbol':name})
                if len(edges)>=limit:return edges
        for imported in anchor.get('imports',[])[:20]:
            modules=re.findall(r'[\w./-]+',imported['text'])
            for module in modules:
                stem=module.rstrip('/').split('/')[-1]
                for target in by_module.get((group(anchor),stem),[])[:2]:
                    if target['documentId']==anchor['documentId'] or target['id'] in seen:continue
                    seen.add(target['id']);edges.append({'from':identifier,'to':target['id'],'kind':'imports','symbol':stem})
                    if len(edges)>=limit:return edges
    return edges
