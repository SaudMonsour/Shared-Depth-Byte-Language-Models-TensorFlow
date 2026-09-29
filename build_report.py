"""Build documented analysis from executed artifacts; never inserts example scores."""
import json, hashlib, html, io, contextlib, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
def read(name): return json.loads((P/name).read_text())
def write(name,obj): (P/name).write_text(json.dumps(obj,indent=2)+'\n')
def main():
    m=read('metrics.json'); models=m['models']; chosen=m['selected_model']; data=read('data_sources.json'); splits=read('splits.json')
    hist=pd.read_csv(P/'training_history.csv'); err=pd.read_csv(P/'error_analysis.csv'); be=pd.read_csv(P/'byte_errors.csv'); freq=pd.read_csv(P/'byte_frequencies.csv')
    checks=read('verification.json')
    for name,r in models.items():
        e=err[err.model==name]; assert len(e)==splits['test']['windows']
        assert abs(e.nll.mean()-r['test']['nll'])<1e-7
        assert abs(e.accuracy.mean()-r['test']['accuracy'])<1e-12
        assert abs(r['test']['perplexity']-2**r['test']['bits_per_byte'])<1e-10
    checks['metric_recalculation']='Passed for all four models from all test-window records'
    for fn in ['standard.weights.npz','shared.weights.npz']:
        with np.load(P/fn,allow_pickle=False) as f:
            arrays=[f[k] for k in f.files]; assert all(np.isfinite(a).all() for a in arrays)
            name='Standard Transformer' if fn.startswith('standard') else 'Shared-depth Transformer'
            assert sum(a.size for a in arrays)==models[name]['parameters']
            checks[fn]={'arrays':len(arrays),'parameters':sum(a.size for a in arrays),'all_values_finite':True}
    write('verification.json',checks)
    figs=P/'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    colors=['#2563eb','#e87924','#64748b','#94a3b8']; names=list(models)
    short={'Standard Transformer':'Standard','Shared-depth Transformer':'Shared depth','Unigram baseline':'Unigram','Bigram baseline':'Bigram'}
    def save(name): plt.tight_layout(); plt.savefig(figs/name,dpi=150,bbox_inches='tight'); plt.close()
    plt.figure(figsize=(8,4.5))
    for i,(name,h) in enumerate(hist.groupby('model',sort=False)):
        plt.plot(h.epoch,h.validation_bpb,'o-',color=colors[i],label=short[name]+' validation')
        plt.plot(h.epoch,h.train_nll/np.log(2),'--',color=colors[i],alpha=.65,label=short[name]+' training')
    plt.xlabel('Epoch'); plt.ylabel('Bits per byte (lower is better)'); plt.title('Learning under the same five-epoch budget'); plt.xticks(range(1,6)); plt.legend(fontsize=9); save('learning-curves.png')
    plt.figure(figsize=(8,4.5)); vals=[models[n]['test']['bits_per_byte'] for n in names]
    bars=plt.bar([short[n] for n in names],vals,color=colors); plt.bar_label(bars,fmt='%.3f',padding=3); plt.ylim(0,max(vals)*1.18); plt.ylabel('Test bits per byte'); plt.title('Held-out next-byte prediction'); save('model-comparison.png')
    plt.figure(figsize=(7,4)); pars=[models[n]['parameters'] for n in names[:2]]
    bars=plt.bar([short[n] for n in names[:2]],pars,color=colors[:2]); plt.bar_label(bars,labels=[f'{p:,}' for p in pars],padding=3); plt.ylim(0,max(pars)*1.2); plt.ylabel('Trainable parameters'); plt.title('Three block applications in both models'); save('parameter-comparison.png')
    f=freq.nlargest(20,'train_count'); plt.figure(figsize=(9,4)); plt.bar([repr(chr(b)) for b in f.byte_id],f.train_count,color='#2563eb'); plt.xticks(rotation=45); plt.ylabel('Training occurrences'); plt.title('Most frequent training bytes'); save('distributions.png')
    plt.figure(figsize=(8,4.5)); neural_losses=err[err.model.isin(names[:2])].nll/np.log(2)
    bins=np.linspace(neural_losses.min(),neural_losses.max(),36)
    for i,n in enumerate(names[:2]): plt.hist(err[err.model==n].nll/np.log(2),bins=bins,alpha=.5,label=short[n],color=colors[i])
    plt.xlabel('Test-window bits per byte'); plt.ylabel('64-byte windows'); plt.title('Prediction difficulty varies across the text'); plt.legend(); save('error-analysis.png')
    b=be[(be.model==chosen)&(be['count']>=100)].nlargest(15,'nll'); plt.figure(figsize=(8,4.5)); plt.bar(b.byte_repr,b.nll/np.log(2),color='#e87924'); plt.xticks(rotation=45); plt.ylabel('Mean bits per target byte'); plt.title('Hardest common bytes: selected model (at least 100 targets)'); save('byte-errors.png')
    reduction=100*(1-pars[1]/pars[0]); gap=models[names[1]]['test']['bits_per_byte']-models[names[0]]['test']['bits_per_byte']
    def display_sample(s):
        return ''.join(ch if ch=='\n' or 32<=ord(ch)<127 else f'\\u{ord(ch):04x}' for ch in s).replace('```','[backticks]')
    sample_previews='\n\n'.join('**'+s['model']+' — first predetermined prompt**\n\n```text\n'+s['prompt']+display_sample(s['generated'])+'\n```' for s in read('generated_samples.json') if s['prompt']=='ROMEO:\n')
    validation='\n'.join(f"| {n} | {r['validation']['bits_per_byte']:.4f} | {r.get('best_epoch','N/A')} |" for n,r in models.items())
    test='\n'.join(f"| {n} | {r['test']['bits_per_byte']:.4f} | {r['test']['perplexity']:.4f} | {100*r['test']['accuracy']:.2f}% |" for n,r in models.items())
    hardest=b.iloc[0]
    source=data['url'].replace('raw.githubusercontent.com','github.com').replace('/6f9487','/blob/6f9487')
    text=f'''# Shared-Depth Byte Language Models in TensorFlow

Train language models from scratch on Shakespeare to measure the accuracy–parameter tradeoff of reusing a Transformer block.

---

## Overview

* **Question:** Can one Transformer block applied three times retain next-byte prediction quality with fewer parameters than three distinct blocks?
* **Task:** Autoregressive byte-level language modeling.
* **Dataset:** {data['bytes']:,} raw bytes; 256 possible byte IDs; 64-byte context windows.
* **Selected Model:** **{chosen}** (test bits per byte = **{models[chosen]['test']['bits_per_byte']:.4f}**, lower is better), selected using validation loss.
* **Baseline Comparison:** Smoothed bigram achieved test bits per byte = {models['Bigram baseline']['test']['bits_per_byte']:.4f}; unigram = {models['Unigram baseline']['test']['bits_per_byte']:.4f}.

This is a small, executed study of an architectural idea relevant to LLMs, not a large general-purpose assistant. It uses TensorFlow without pretrained weights, external LLM calls, or a learned tokenizer.

---

## Key Results

### Training Validation

One contiguous validation partition is used instead of cross-validation. Fold standard deviations are not applicable.

| Model | Validation bits per byte | Selected epoch |
| :--- | ---: | ---: |
{validation}

### Final Holdout Evaluation ({splits['test']['target_bytes']:,} Target Bytes)

| Model | Bits per byte ↓ | Byte perplexity ↓ | Next-byte accuracy ↑ |
| :--- | ---: | ---: | ---: |
{test}

Sharing reduces trainable parameters from **{pars[0]:,} to {pars[1]:,} ({reduction:.1f}%)**. Its test BPB difference relative to the standard model is **{gap:+.4f}**; a positive difference means worse prediction. Both execute three attention/FFN applications, so fewer parameters do not imply proportionally less computation. The observed training wall times include validation and checkpoint writing and are not controlled speed benchmarks.

### Architecture and Evaluation Protocol

Both models have width 64, four attention heads, feed-forward width 128, learned position embeddings, pre-layer normalization, causal attention, tied input/output embeddings, and dropout 0.1. The standard model owns three distinct blocks; the shared-depth model invokes one block three times. It has no adaptive halting or depth embeddings.

The raw corpus is split 80%/10%/10% in source order **before** constructing windows. No window crosses a partition boundary. Each complete window supplies 64 next-byte targets, with stride 64 and reset context; targets do not overlap within a partition. The first byte supplies context only, and incomplete tails are excluded. This is a custom split, not an official benchmark split.

Each model receives five epochs, batch size 64, Adam learning rate 0.001, global gradient clipping at 1.0, and fixed seed 42. Validation loss chooses checkpoints and the selected architecture before the test partition is scored. Both prespecified architectures and two count baselines are reported on identical test targets. Count baselines use training text only and additive smoothing 0.1. BPB is mean negative log probability divided by ln(2); byte perplexity is exp(mean NLL).

Training curves average losses while weights are changing and dropout is active; validation is measured after each epoch with dropout disabled. Their difference is therefore not a pure estimate of the generalization gap. Checkpoint reload and a future-token perturbation check verify identical inference and causal masking on saved examples; all reported test losses are independently recalculated from window records.

---

## Data Summary

* **Source:** [Tiny Shakespeare, curated by Andrej Karpathy]({source}), pinned source commit `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`.
* **Data Quality:** {data['bytes']:,} bytes fully decoded as UTF-8; {data['distinct_bytes']} distinct byte values. No imputation, deduplication, or text normalization. Repeated phrases are retained because they are part of the text.
* **Partitions:** {splits['train']['windows']:,} training windows, {splits['validation']['windows']:,} validation windows, and {splits['test']['windows']:,} test windows. Exact offsets and unused tails appear in `splits.json`.
* **Integrity:** All source bytes match the pinned Git blob SHA; SHA-256 is recorded in `data_sources.json`. The corpus is referenced rather than republished.
* **Interpretability:** Permutation feature importance and tabular correlations are not appropriate here. `byte_errors.csv` reports target-byte difficulty; it is not a causal explanation.

### Error Analysis and Limitations

The selected model's hardest target among bytes with at least 100 test occurrences is `{hardest.byte_repr}`, with mean {hardest.nll/np.log(2):.3f} bits and {int(hardest['count']):,} occurrences. Per-window losses and accuracies are retained for all four models in `error_analysis.csv`.

All six generated continuations are preserved in `generated_samples.json`: three predetermined prompts per neural model, 256 new bytes each, temperature 0.8, and seeds 42–44. They are model outputs, not curated examples or evidence of reasoning. Raw output bytes are also retained as hexadecimal strings. Byte prediction does not guarantee grammatical or meaningful generation.

This is one seed, a short context, one small literary corpus, and a fixed short budget. Contiguous partitions can differ in characters and style, while repeated phrases may cross partitions; this is not a held-out-author or held-out-play evaluation. No statistical significance, general knowledge, reasoning, multilingual ability, or large-scale efficiency claim is made. A 256-byte vocabulary accepts arbitrary UTF-8 input but does not demonstrate multilingual learning on this predominantly English corpus.

### Actual Generated Text

The first predetermined prompt from each model is shown below without selecting for quality. Non-printable/non-ASCII output is escaped for display; raw bytes are preserved in the JSON artifact. These continuations can contain invented words, broken syntax, and inconsistent meaning.

{sample_previews}

---

## Visualizations

| Learning curves | Holdout performance |
| :---: | :---: |
| ![Learning curves](figures/learning-curves.png) | ![Model comparison](figures/model-comparison.png) |

| Parameter count | Training-byte distribution |
| :---: | :---: |
| ![Parameters](figures/parameter-comparison.png) | ![Distributions](figures/distributions.png) |

| Window-level error distribution | Difficult target bytes |
| :---: | :---: |
| ![Error analysis](figures/error-analysis.png) | ![Byte errors](figures/byte-errors.png) |

---

## Repository Structure

```text
├── figures/                   # Six measured diagnostic and comparison plots
├── analysis.ipynb             # Executed artifact-analysis walkthrough
├── train.py                   # Models, training, evaluation, and generation
├── build_report.py            # Artifact checks, figures, and documentation
├── generate.py                # Generate text from either saved checkpoint
├── study_config.json          # Fixed experiment choices
├── requirements.txt           # Direct Python dependencies
├── environment.lock.txt       # Full executed dependency versions
├── audit.json                 # Environment, timestamps, and code hashes
├── data_sources.json          # Pinned source URL and integrity hashes
├── splits.json                # Source offsets and evaluated byte counts
├── byte_frequencies.csv        # Training counts for all 256 byte IDs
├── byte_errors.csv             # Neural error rates by target byte
├── error_analysis.csv          # Test-window metrics for every model
├── training_history.csv        # Actual per-epoch losses
├── generated_samples.json      # All predetermined continuations
├── standard.weights.npz        # Standard model's selected weights
├── shared.weights.npz          # Shared model's selected weights
├── selection.json              # Validation-only checkpoint/model selection
├── metrics.json                # Full validation and test results
├── verification.json           # Causality, reload, and metric checks
├── SOURCE.md                   # Dataset and research attribution
├── LEARNING_NOTES.md           # Unfilled human review prompts
├── report.html                 # Report referencing the figures directory
├── repository_metadata.json    # Suggested name, About, and topics
├── project.json                # Completed-study manifest and artifact hashes
└── README.md
```

## Research Context and Attribution

Depth sharing is inspired by [Universal Transformers](https://arxiv.org/abs/1807.03819), and raw-byte modeling by [ByT5](https://arxiv.org/abs/2105.13626). These ideas are established research; this study does not claim to invent them or reproduce either full architecture. Here the experiment is a small causal decoder with fixed repeated depth.

Automated implementation, training, analysis, and documentation by OpenAI Codex for Saud Alotaibi. Human review is pending; personal learning answers are intentionally unfilled.
'''
    (P/'README.md').write_text(text)
    write('repository_metadata.json',dict(suggested_repo_name='tensorflow-shared-depth-byte-lm',about='From-scratch TensorFlow byte language models: compare shared-depth and standard Transformers on Shakespeare with verified metrics and checkpoints.',topics=['tensorflow','language-modeling','transformer','deep-learning','byte-level','parameter-sharing','nlp']))
    source_text=f'''# Sources and provenance

Corpus: Tiny Shakespeare, distributed by Andrej Karpathy in char-rnn.
Pinned raw URL: {data['url']}
Git blob SHA-1: {data['git_blob_sha']}
SHA-256: {data['sha256']}

The complete downloaded byte stream was checked against the source Git blob, decoded strictly as UTF-8, and counted before splitting. No data snapshot is included.

The underlying Shakespeare works are public-domain historical texts. The upstream repository tree has no standalone license file, and the Hugging Face dataset card leaves licensing metadata unspecified; no MIT license is asserted for the corpus compilation. Reference: https://huggingface.co/datasets/karpathy/tiny_shakespeare/blob/main/README.md .

Architecture references:
- Dehghani et al., Universal Transformers (ICLR 2019): https://arxiv.org/abs/1807.03819
- Xue et al., ByT5: Towards a token-free future with pre-trained byte-to-byte models (TACL 2022): https://arxiv.org/abs/2105.13626

The implementation is independently written TensorFlow code. It uses fixed-depth causal weight sharing, not adaptive computation or the ByT5 encoder-decoder architecture. Training uses no pretrained weights or paid language-model API.
'''
    (P/'SOURCE.md').write_text(source_text)
    (P/'LEARNING_NOTES.md').write_text('# Human learning review\n\n## Why might depth sharing help or hurt?\n\n## What does bits per byte measure?\n\n## Which limitation matters most?\n\n## What would I change in a follow-up?\n\n')
    # Execute simple notebook cells in-process; stored outputs are captured stdout.
    codes=["import json\nfrom pathlib import Path\nimport pandas as pd\np = Path('.')\nm = json.loads((p / 'metrics.json').read_text())\nprint('Selected using validation:', m['selected_model'])", "rows = [{'model': k, **v['test']} for k, v in m['models'].items()]\nprint(pd.DataFrame(rows).to_string(index=False))", "e = pd.read_csv(p / 'error_analysis.csv')\nfor name, v in m['models'].items():\n    assert abs(e[e.model == name].nll.mean() - v['test']['nll']) < 1e-7\nprint('All test NLL values match saved window records.')", "s = json.loads((p / 'generated_samples.json').read_text())\nfor sample in s:\n    print(sample['model'], repr(sample['prompt']), repr(sample['generated']))"]
    import os
    old=os.getcwd(); os.chdir(P); namespace={}; cells=[dict(cell_type='markdown',metadata={},source=['# Shared-depth byte language model analysis\n','This notebook inspects executed training artifacts; `train.py` contains the complete training pipeline.'])]
    for i,code in enumerate(codes,1):
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf): exec(compile(code,f'notebook-cell-{i}','exec'),namespace)
        cells.append(dict(cell_type='code',metadata={},execution_count=i,source=code.splitlines(keepends=True),outputs=[dict(output_type='stream',name='stdout',text=buf.getvalue().splitlines(keepends=True))]))
    os.chdir(old)
    for i,cell in enumerate(cells): cell['id']=f'analysis-{i}'
    write('analysis.ipynb',dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'},'execution_method':'Python in-process execution of each code cell with stdout capture'},nbformat=4,nbformat_minor=5))
    # A compact HTML report, with linked image files rather than duplicate image bytes.
    body='<h1>Shared-Depth Byte Language Models in TensorFlow</h1><p>Automated study; human review pending.</p><p>'+html.escape(f'Selected: {chosen}. Parameter reduction: {reduction:.1f}%. Shared minus standard test BPB: {gap:+.4f}.')+'</p>'
    body+=pd.DataFrame([{'model':n,**r['test']} for n,r in models.items()]).to_html(index=False,float_format=lambda v:f'{v:.4f}')
    body+='<p>Five fixed epochs; one seed; contiguous 80/10/10 split; no pretrained weights. See README.md for full protocol and limitations.</p>'
    for fn in sorted(figs.glob('*.png')): body+=f'<figure><img src="figures/{fn.name}" alt="{html.escape(fn.stem)}"></figure>'
    (P/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared-depth language modeling study</title><style>body{max-width:1000px;margin:40px auto;font:17px system-ui;padding:20px}img{max-width:100%}td,th{padding:10px;text-align:left}figure{margin:28px 0}</style>'+body+'</html>')
    print(json.dumps({'selected':chosen,'parameter_reduction_percent':reduction,'test_bpb_gap':gap,'checks':checks},indent=2))
if __name__=='__main__': main()
