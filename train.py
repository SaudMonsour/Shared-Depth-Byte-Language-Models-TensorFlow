"""Fixed-budget byte language modeling experiment; run with Python 3.12."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json, hashlib, time, platform, urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
P=Path(__file__).resolve().parent
C=dict(seed=42,context=64,width=64,heads=4,ffn=128,depth=3,dropout=0.1,epochs=5,batch_size=64,learning_rate=0.001,baseline_alpha=0.1,prompts=['ROMEO:\n','JULIET:\n','To be, or not to be'],generation_bytes=256,temperature=0.8)
URL='https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt'
def save(name,obj): (P/name).write_text(json.dumps(obj,indent=2)+'\n')
def data():
    cache=P.parent.parent/'tinyshakespeare.txt'
    if not cache.exists(): cache.write_bytes(urllib.request.urlopen(URL).read())
    raw=cache.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9'
    raw.decode('utf-8',errors='strict')
    a=np.frombuffer(raw,dtype=np.uint8).astype('int32'); n=len(a)
    boundaries=[0,int(n*.8),int(n*.9),n]; sets={}; splits={}
    for name,lo,hi in zip(['train','validation','test'],boundaries[:-1],boundaries[1:]):
        chunk=a[lo:hi]; k=(len(chunk)-1)//C['context']; used=k*C['context']
        sets[name]=(chunk[:used].reshape(k,64),chunk[1:used+1].reshape(k,64))
        splits[name]=dict(start=lo,stop=hi,windows=k,target_bytes=used,unused_tail_bytes=len(chunk)-used-1)
    return raw,a,sets,splits
class Block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(); self.n1=tf.keras.layers.LayerNormalization(); self.n2=tf.keras.layers.LayerNormalization()
        self.att=tf.keras.layers.MultiHeadAttention(num_heads=4,key_dim=16,dropout=.1)
        self.ff=tf.keras.Sequential([tf.keras.layers.Dense(128,activation='gelu'),tf.keras.layers.Dense(64),tf.keras.layers.Dropout(.1)])
    def call(self,x,training=False):
        z=self.n1(x); x=x+self.att(z,z,use_causal_mask=True,training=training)
        return x+self.ff(self.n2(x),training=training)
class LM(tf.keras.Model):
    def __init__(self,shared):
        super().__init__(); self.shared=shared; self.emb=tf.keras.layers.Embedding(256,64); self.pos=tf.keras.layers.Embedding(64,64)
        self.blocks=[Block() for _ in range(1 if shared else 3)]; self.norm=tf.keras.layers.LayerNormalization()
    def call(self,t,training=False):
        x=self.emb(t)+self.pos(tf.range(tf.shape(t)[1]))
        for i in range(3): x=self.blocks[0 if self.shared else i](x,training=training)
        return tf.matmul(self.norm(x),self.emb.embeddings,transpose_b=True)
def dataset(x,y,shuffle=False):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    if shuffle: ds=ds.shuffle(len(x),seed=42,reshuffle_each_iteration=True)
    opt=tf.data.Options(); opt.threading.private_threadpool_size=2; opt.experimental_deterministic=True
    return ds.batch(64).with_options(opt).prefetch(1)
def evaluate(model,x,y):
    loss=[]; correct=[]
    for start in range(0,len(x),64):
        z=model(x[start:start+64],training=False)
        loss.append(tf.keras.losses.sparse_categorical_crossentropy(y[start:start+64],z,from_logits=True).numpy())
        correct.append(z.numpy().argmax(-1)==y[start:start+64])
    return np.concatenate(loss),np.concatenate(correct)
def summary(loss,correct):
    nll=float(np.mean(loss,dtype=np.float64)); return dict(nll=nll,bits_per_byte=nll/np.log(2),perplexity=float(np.exp(nll)),accuracy=float(np.mean(correct)))
def main():
    save('study_config.json',C); start=datetime.now(ZoneInfo('Asia/Riyadh')).isoformat()
    raw,a,sets,splits=data(); save('splits.json',splits)
    save('data_sources.json',dict(url=URL,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha='7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9',distinct_bytes=int(len(np.unique(a))),utf8_valid=True,snapshot_stored=False))
    histories=[]; models={}; results={}; checks={}
    for name,shared in [('Standard Transformer',False),('Shared-depth Transformer',True)]:
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42)
        model=LM(shared); model(sets['train'][0][:1]); key='shared' if shared else 'standard'
        model.compile(optimizer=tf.keras.optimizers.Adam(.001,global_clipnorm=1.),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        best=float('inf'); best_epoch=0; tic=time.perf_counter()
        train_ds=dataset(*sets['train'],shuffle=True)
        for epoch in range(1,6):
            hist=model.fit(train_ds,epochs=1,verbose=0)
            vl,vc=evaluate(model,*sets['validation']); vm=summary(vl,vc)
            row=dict(model=name,epoch=epoch,train_nll=float(hist.history['loss'][0]),validation_nll=vm['nll'],validation_bpb=vm['bits_per_byte'])
            histories.append(row); print(json.dumps(row),flush=True)
            if vm['nll']<best:
                best=vm['nll']; best_epoch=epoch
                np.savez(P/f'{key}.weights.npz',**{f'w{i}':v for i,v in enumerate(model.get_weights())})
            pd.DataFrame(histories).to_csv(P/'training_history.csv',index=False)
        elapsed=time.perf_counter()-tic
        with np.load(P/f'{key}.weights.npz',allow_pickle=False) as f: model.set_weights([f[f'w{i}'] for i in range(len(f.files))])
        models[name]=model
        results[name]=dict(parameters=model.count_params(),best_epoch=best_epoch,training_seconds=elapsed,validation=summary(*evaluate(model,*sets['validation'])))
    selected=min(results,key=lambda k:results[k]['validation']['nll'])
    save('selection.json',dict(selected_model=selected,rule='Minimum validation NLL; checkpoints and model selected before test evaluation',models=results))
    x,y=sets['test']; rows=[]; byte_rows=[]; samples=[]
    for name,model in models.items():
        loss,correct=evaluate(model,x,y); results[name]['test']=summary(loss,correct)
        for i in range(len(x)): rows.append(dict(model=name,window=i,source_target_start=splits['test']['start']+i*64+1,nll=float(loss[i].mean(dtype=np.float64)),accuracy=float(correct[i].mean())))
        for b in np.unique(y):
            mask=y==b; byte_rows.append(dict(model=name,byte_id=int(b),byte_repr=repr(chr(b)),count=int(mask.sum()),nll=float(loss[mask].mean(dtype=np.float64)),accuracy=float(correct[mask].mean())))
        z=model(x[:2],training=False).numpy(); altered=x[:2].copy(); altered[:,32:]=(altered[:,32:]+1)%256
        causal=float(np.max(np.abs(z[:,:32]-model(altered,training=False).numpy()[:,:32])))
        clone=LM(model.shared); clone(x[:1]); key='shared' if model.shared else 'standard'
        with np.load(P/f'{key}.weights.npz',allow_pickle=False) as f: clone.set_weights([f[f'w{i}'] for i in range(len(f.files))])
        reload_error=float(np.max(np.abs(z-clone(x[:2],training=False).numpy())))
        assert causal<1e-6 and reload_error<1e-6
        checks[name]=dict(causal_max_logit_error=causal,reload_max_logit_error=reload_error)
        for j,prompt in enumerate(C['prompts']):
            rng=np.random.default_rng(42+j); tokens=list(prompt.encode()); generated=[]
            for _ in range(256):
                logits=model(np.array([tokens[-64:]],dtype=np.int32),training=False).numpy()[0,-1].astype('float64')/.8
                p=np.exp(logits-logits.max()); p/=p.sum(); b=int(rng.choice(256,p=p)); tokens.append(b); generated.append(b)
            samples.append(dict(model=name,prompt=prompt,seed=42+j,temperature=.8,generated=bytes(generated).decode('utf8',errors='replace'),raw_hex=bytes(generated).hex()))
    tr=a[:splits['train']['stop']]; counts=np.bincount(tr,minlength=256)+.1; uni=counts/counts.sum()
    counts2=np.full((256,256),.1); np.add.at(counts2,(tr[:-1],tr[1:]),1); bi=counts2/counts2.sum(1,keepdims=True)
    for name,probs,pred in [('Unigram baseline',uni[y],np.full_like(y,uni.argmax())),('Bigram baseline',bi[x,y],bi.argmax(1)[x])]:
        loss=-np.log(probs); correct=pred==y; results[name]=dict(test=summary(loss,correct))
        vx,vy=sets['validation']; vp=uni[vy] if name.startswith('Unigram') else bi[vx,vy]; vc=np.full_like(vy,uni.argmax()) if name.startswith('Unigram') else bi.argmax(1)[vx]
        results[name]['validation']=summary(-np.log(vp),vc==vy)
        for i in range(len(x)): rows.append(dict(model=name,window=i,source_target_start=splits['test']['start']+i*64+1,nll=float(loss[i].mean()),accuracy=float(correct[i].mean())))
    pd.DataFrame(rows).to_csv(P/'error_analysis.csv',index=False); pd.DataFrame(byte_rows).to_csv(P/'byte_errors.csv',index=False)
    pd.DataFrame({'byte_id':np.arange(256),'train_count':np.bincount(tr,minlength=256)}).to_csv(P/'byte_frequencies.csv',index=False)
    save('generated_samples.json',samples); save('metrics.json',dict(selected_model=selected,models=results)); save('verification.json',checks)
    save('audit.json',dict(started_at=start,finished_at=datetime.now(ZoneInfo('Asia/Riyadh')).isoformat(),python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,seed=42,device='CPU',deterministic_ops=True,config_sha256=hashlib.sha256((P/'study_config.json').read_bytes()).hexdigest(),train_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__': main()
