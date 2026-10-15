"""Author model archives, verified downloads and user-owned imports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.request
import zipfile

CATALOG={
    'illustration-denoise':dict(scale=4,name='IllustrationJaNai V3 · 去噪 · DAT2',
        url='https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3denoise.zip',
        sha256='6c26828ad8e142c9e9c7181451257f3f56a0f5c7f897d9c7878ce898404f53e5',bytes=166680846,
        source='https://github.com/the-database/MangaJaNai/releases/tag/3.0.0'),
    'illustration-detail':dict(scale=4,name='IllustrationJaNai V3 · 细节 · DAT2',
        url='https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3detail.zip',
        sha256='27749962bc934779f867730cdb10617e3d7475313f920a7a966c56b75dbdbeda',bytes=318641534,
        source='https://github.com/the-database/MangaJaNai/releases/tag/3.0.0'),
    'animesharp':dict(scale=2,name='AnimeSharp V4 · RCAN',
        url='https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/1a9339b/2x-AnimeSharpV4_RCAN.safetensors',
        sha256='6470bb91d6622d6acdff81132c1a8615b961b919ce2b9a01ce993378500cfbe1',bytes=31053198,
        source='https://huggingface.co/Kim2091/2x-AnimeSharpV4'),
}

def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()

class ModelStore:
    def __init__(self,root=None):
        self.root=Path(root or Path(os.environ.get('LUNARC_DATA_DIR') or
            Path(os.environ.get('LOCALAPPDATA') or Path.home())/'Lunarc')/'image-models')
        self.lock=threading.Lock();self.progress=dict(busy=None,received=0,total=0,error=None)

    def status(self):
        models={}
        for key,entry in CATALOG.items():
            try:
                record=json.loads((self.root/(key+'.json')).read_text(encoding='utf-8'))
                installed=(self.root/record['file']).is_file()
            except (OSError,ValueError,KeyError):record={};installed=False
            models[key]=dict(entry,installed=installed,record=record)
        with self.lock:return dict(self.progress,models=models)

    def resolve(self,key):
        if key not in CATALOG:raise ValueError('未知图片放大模型')
        try:
            record=json.loads((self.root/(key+'.json')).read_text(encoding='utf-8'))
            file=self.root/record['file']
            if file.resolve().parent!=self.root.resolve() or digest(file)!=record['sha256']:
                raise ValueError('图片模型校验失败，请重新下载或导入')
            return file
        except (OSError,KeyError,ValueError) as error:
            raise ValueError(f'{CATALOG[key]["name"]} 尚未安装或缓存已损坏，请在磁贴中下载／导入模型') from error

    def import_model(self,key,path,*,official=False):
        if key not in CATALOG:raise ValueError('未知图片放大模型')
        source=Path(path)
        if source.suffix.lower() not in ('.pth','.safetensors') or not source.is_file():
            raise ValueError('请选择 .pth 或 .safetensors 模型文件')
        if not 0<source.stat().st_size<512*1024**2:raise ValueError('模型文件大小无效')
        self.root.mkdir(parents=True,exist_ok=True)
        name=key+source.suffix.lower();target=self.root/name
        with tempfile.NamedTemporaryFile(dir=self.root,delete=False) as temporary:
            staged=Path(temporary.name)
            with source.open('rb') as stream:shutil.copyfileobj(stream,temporary)
        try:
            sha=digest(staged);os.replace(staged,target)
            record=dict(file=name,sha256=sha,official=official,source=CATALOG[key]['source'] if official else 'user import')
            metadata=self.root/(key+'.json.tmp');metadata.write_text(json.dumps(record),encoding='utf-8')
            os.replace(metadata,self.root/(key+'.json'))
        finally:staged.unlink(missing_ok=True)
        return self.status()

    def start(self,key):
        if key not in CATALOG:raise ValueError('未知图片放大模型')
        with self.lock:
            if self.progress['busy']:return self.status_unlocked()
            self.progress=dict(busy=key,received=0,total=CATALOG[key]['bytes'],error=None)
        threading.Thread(target=self._download,args=(key,),daemon=True).start()
        return self.status()

    def status_unlocked(self):return dict(self.progress)

    def _download(self,key):
        entry=CATALOG[key]
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                package=Path(directory)/('model.zip' if key.startswith('illustration') else 'model.safetensors')
                request=urllib.request.Request(entry['url'],headers={'User-Agent':'Lunarc/1.1'})
                with urllib.request.urlopen(request,timeout=60) as response,package.open('wb') as output:
                    received=0
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        received+=len(chunk)
                        if received>entry['bytes']:raise ValueError('模型下载长度不匹配')
                        output.write(chunk)
                        with self.lock:self.progress['received']=received
                if received!=entry['bytes'] or digest(package)!=entry['sha256']:
                    raise ValueError('模型下载 SHA-256／长度不匹配')
                if package.suffix=='.zip':
                    with zipfile.ZipFile(package) as archive:
                        names=[name for name in archive.namelist() if 'DAT2' in name and name.lower().endswith(('.pth','.safetensors'))]
                        if len(names)!=1:raise ValueError('官方模型压缩包的 DAT2 布局不匹配')
                        info=archive.getinfo(names[0])
                        if info.file_size>256*1024**2:raise ValueError('模型解压大小无效')
                        model=Path(directory)/('model'+Path(names[0]).suffix)
                        with archive.open(names[0]) as source,model.open('wb') as output:shutil.copyfileobj(source,output)
                else:model=package
                self.import_model(key,model,official=True)
        except Exception as error:
            with self.lock:self.progress['error']=str(error)
        finally:
            with self.lock:self.progress['busy']=None

store=ModelStore()

def selection_key(settings):
    from native_effects import stages,stage_kind,stage_settings
    values=[]
    for stage in stages(settings):
        kind=stage_kind(stage)
        if kind not in ('illustration','animesharp'):continue
        key='illustration-'+stage_settings(settings,stage).get('image_model_variant','denoise') if kind=='illustration' else 'animesharp'
        try:stamp=(store.root/(key+'.json')).stat().st_mtime_ns
        except OSError:stamp=None
        values.append((stage,key,stamp))
    return tuple(values)
