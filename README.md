# Provably Learning Object-Centric Representations [ICML 2023]
Official code for the paper [Provably Learning Object-Centric Representations](https://arxiv.org/abs/2305.14229).

![Problem Setup](problem_setup.png)

## Synthetic Data
The experiments on synthetic non-image data in Section 5.1 of the paper can be run with the following command:
```
python train_model.py --data synth --num_slots --lam --dependent --lr 1e-3 --num_iters 115000
```

## Image Data
To run the experiments on image data in Section 5.2 of the paper, you should first generate a sprites dataset using the following command:
```
python data/generators/sprites_data_gen.py --max_objects 4
```
To run experiments with the additive autoencoder model on this data, use the command:
```
python train_model.py --data spriteworld --encoder monolithic --decoder baseline --num_slots 4 --inf_slot_dim 16 --num_iters 500000
```
To run experiments with the Slot Attention autoencoder model on this data, use the command:
```
python train_model.py --data spriteworld --encoder slot-attention --decoder spatial-broadcast --num_slots 4 --inf_slot_dim 16 --num_iters 500000
```

To run experiments with the MONet model on this data, use the command:
```
python train_model.py --data spriteworld --encoder monet --decoder monet --num_slots 4 --inf_slot_dim 16 --num_iters 500000
```

## BibTeX

If you make use of this code in your own work, please cite our paper:
```
@inproceedings{Brady2023ProvablyLO,
  title = 	 {Provably Learning Object-Centric Representations},
  author =       {Brady, Jack and Zimmermann, Roland S. and Sharma, Yash and Sch\"{o}lkopf, Bernhard and Von K\"{u}gelgen, Julius and Brendel, Wieland},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {3038--3062},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR}
}
```
