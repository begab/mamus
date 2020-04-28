# MaMuS (Massively Multilingual Sparse Word Representations)

If you want to use the pretrained multilingual sparse word embeddings that served as the basis for the experiments in the ICLR2020 paper [Massively Multilingual Sparse Word Representations](https://openreview.net/forum?id=HyeYTgrFPB) you can do it so from the above [link](http://www.inf.u-szeged.hu/~berendg/docs/multiling75_isometric_-1_fasttext_cbow100_1200_0.1_unit-sparse0cos_pos.vec.gz).

## Colab demo

You can also familiarize yourself with the kind of sparse we trained in this [Colab notebook](https://colab.research.google.com/drive/1sMtCrGepORHG5t3HI79G6mdA2Npjc9W4)

## Training your own MaMuS word representations

```
git clone git@github.com:begab/mamus.git
cd mamus

conda create --name mamus python==3.6.3
source activate mamus  
pip install -r requirements.txt && conda install -c conda-forge python-spams  

wget http://www.inf.u-szeged.hu/~berendg/docs/fasttext_cbow100_{en,fr}.vec.gz

python mamus.py --embedding-mode fasttext_cbow100 --dictionary-fallback --dictionary-file dictionaries/massively_multiling/parallel.fwdxbwd-dict.fr-en.gz --source-embedding fasttext_cbow100_en.vec.gz --target-embedding fasttext_cbow100_fr.vec.gz --out-path en_fr_mamus.vec > mamus.log 2>&1 &
```

## Bibtex for the publication

```
@inproceedings{  
  berend2020massively,  
  title={Massively Multilingual Sparse Word Representations},  
  author={G{\'a}bor Berend},  
  booktitle={International Conference on Learning Representations},  
  year={2020},  
  url={https://openreview.net/forum?id=HyeYTgrFPB}  
}
```
