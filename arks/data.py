import os
from arks.utils import get_str_len

class ArksData:
    def __init__(self,task,data_dir,knowledge,retrieval_tokenizer):
        self.task = task
        self.data_dir = data_dir
        self.doc_split_token = '##'
        self.knowledge = knowledge
        self.retrieval_tokenizer = retrieval_tokenizer

    def get_docs(self):
        docs = {}
        total = 0
        empty = 0
        for file in os.listdir(os.path.join(self.data_dir,self.task,'docs')):
            if not file.endswith('.md'):
                continue
            total += 1
            with open(os.path.join(self.data_dir,self.task,'docs',file)) as f:
                content = f.read()
            new_content = ''
            if self.task=='Pony':
                if 'code_snippets' in self.knowledge and not 'doc' in self.knowledge:
                    import re
                    indices = [m.start() for m in re.finditer('```', content)]
                    for i in range(0, len(indices) - 1, 2):
                        cur_snippet = content[indices[i]:indices[i + 1]].replace('```pony', '').replace('```', '')
                        cur_snippet = f"```pony\n{cur_snippet}\n```"
                        new_content += cur_snippet + '\n'
                elif 'code_snippets' not in self.knowledge and 'doc' in self.knowledge:
                    if '```' in content:
                        import re
                        indices = [0]
                        indices += [m.start() for m in re.finditer('```', content)]
                        for i in range(0,len(indices)-1,2):
                            new_content += content[indices[i]:indices[i+1]].replace('```','')
                    else:
                        new_content = content
                else:
                    new_content = content
            elif self.task=='Ring':
                if 'code_snippets' in self.knowledge and not 'doc' in self.knowledge:
                    if 'example' in content.lower():
                        new_content = content[content.lower().index('example'):]
                elif 'code_snippets' not in self.knowledge and 'doc' in self.knowledge:
                    if 'example' in content.lower():
                        new_content = content[:content.lower().index('example')]
                    else:
                        new_content = content
                else:
                    new_content = content
            elif self.task=='ScipyM' or self.task=='TensorflowM':
                if 'code_snippets' in self.knowledge and not 'doc' in self.knowledge:
                    content_parts = content.split('#')
                    new_content_parts = []
                    for p in content_parts:
                        if 'example' in p.lower() or '>>>' in p:
                            new_content_parts.append(p)
                    new_content = '#'.join(new_content_parts)
                elif 'code_snippets' not in self.knowledge and 'doc' in self.knowledge:
                    content_parts = content.split('#')
                    new_content_parts = []
                    for p in content_parts:
                        if not 'example' in p.lower():
                            new_content_parts.append(p)
                    new_content = '#'.join(new_content_parts)
                else:
                    new_content = content
            content = new_content
            if len(content)==0:
                empty += 1
                continue
            if self.retrieval_tokenizer is not None and get_str_len(content)>500:
                parts = content.split(self.doc_split_token)
                cur_doc = []
                for p in parts:
                    if p=='':
                        continue
                    section = p
                    cur_doc.append(section)
                docs[file] = cur_doc
            else:
                docs[file] = [content]
        return docs
    
