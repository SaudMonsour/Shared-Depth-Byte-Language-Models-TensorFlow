"""Generate from a published checkpoint. Example: python generate.py --prompt 'ROMEO:'"""
import argparse
import numpy as np
from train import LM, P

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['standard','shared'],default='shared')
    parser.add_argument('--prompt',default='ROMEO:\n')
    parser.add_argument('--bytes',type=int,default=256)
    parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if not args.prompt or args.temperature<=0 or args.bytes<0: parser.error('Use a nonempty prompt, positive temperature, and nonnegative byte count.')
    model=LM(args.model=='shared'); model(np.array([[0]],dtype=np.int32))
    with np.load(P/f'{args.model}.weights.npz',allow_pickle=False) as f:
        model.set_weights([f[f'w{i}'] for i in range(len(f.files))])
    tokens=list(args.prompt.encode('utf8')); rng=np.random.default_rng(args.seed)
    for _ in range(args.bytes):
        logits=model(np.array([tokens[-64:]],dtype=np.int32),training=False).numpy()[0,-1].astype('float64')/args.temperature
        p=np.exp(logits-logits.max()); p/=p.sum(); tokens.append(int(rng.choice(256,p=p)))
    print(bytes(tokens).decode('utf8',errors='replace'))

if __name__=='__main__': main()
