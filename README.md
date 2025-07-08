<img src="./fig1.png" width="500px"></img>

## Strassen Attention

Implementation of [Strassen attention](https://arxiv.org/abs/2501.19215), from Kozachinskiy et al. of [National Center of AI](https://cenia.cl/) in Chile 🇨🇱

## Install

```shell
$ pip install strassen-attention
```

## Usage

```python
import torch
from strassen_attention import strassen_attend

q = torch.randn(1, 8, 32, 16)
k = torch.randn(1, 8, 32, 16)
v = torch.randn(1, 8, 32, 16)

attended = strassen_attend(
    q,
    k,
    k.clone(),
    v,
    v.clone()
)

assert attended.shape == q.shape
```

For the multi-head attention module

```python
import torch
from strassen_attention.strassen_mha import StrassenMHA

mha = StrassenMHA(dim = 512, causal = True)

tokens = torch.randn(1, 256, 512)

assert mha(tokens).shape == tokens.shape
```

Strassen attention transformer

```python
import torch
import torch
from strassen_attention.strassen_transformer import StrassenTransformer

transformer = StrassenTransformer(dim = 512, depth = 4)

x = torch.randn(1, 16 * 16, 512)
assert transformer(x).shape == x.shape
```

## Citations

```bibtex
@misc{kozachinskiy2025strassenattentionunlockingcompositional,
    title   = {Strassen Attention: Unlocking Compositional Abilities in Transformers Based on a New Lower Bound Method}, 
    author  = {Alexander Kozachinskiy and Felipe Urrutia and Hector Jimenez and Tomasz Steifer and Germán Pizarro and Matías Fuentes and Francisco Meza and Cristian B. Calderon and Cristóbal Rojas},
    year    = {2025},
    eprint  = {2501.19215},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2501.19215}, 
}
```

```bibtex
@article{Peng2024OnLO,
    title     = {On Limitations of the Transformer Architecture},
    author    = {Binghui Peng and Srini Narayanan and Christos Papadimitriou},
    journal   = {ArXiv},
    year      = {2024},
    volume    = {abs/2402.08164},
    url       = {https://api.semanticscholar.org/CorpusID:267636545}
}
```
