# MTKD


Multi-teacher Knowledge Distillation for End-to-End Text Image Machine Translation

The official repository for ICDAR 2023 Conference paper: 

- **Cong Ma**, Yaping Zhang, Mei Tu, Yang Zhao, Yu Zhou, and Chengqing Zong. **Multi-Teacher Knowledge Distillation for End-to-End Text Image Machine Translation**. In The 17th Document Analysis and Recognition (ICDAR 2023), San José, California, USA. August 21-26, 2023. pp. 484–501, Cham. Springer Nature Switzerland. (**Oral Paper**) [arXiv](https://arxiv.org/abs/2305.05226), [Springer_Link](https://link.springer.com/chapter/10.1007/978-3-031-41676-7_28)



## 1. Introduction

Text image machine translation (TIMT) has been widely used in various real-world applications, which translates source language texts in images into another target language sentence. Existing methods on TIMT are mainly divided into two categories: the recognition-then- translation pipeline model and the end-to-end model. However, how to transfer knowledge from the pipeline model into the end-to-end model remains an unsolved problem. In this paper, we propose a novel Multi- Teacher Knowledge Distillation (MTKD) method to effectively distillate knowledge into the end-to-end TIMT model from the pipeline model. Specifically, three teachers are utilized to improve the performance of the end-to-end TIMT model. The image encoder in the end-to-end TIMT model is optimized with the knowledge distillation guidance from the recognition teacher encoder, while the sequential encoder and decoder are improved by transferring knowledge from the translation sequential and decoder teacher models. Furthermore, both token and sentence-level knowledge distillations are incorporated to better boost the translation performance. Extensive experimental results show that our proposed MTKD effectively improves the text image translation performance and outperforms existing end-to-end and pipeline models with fewer parameters and less decoding time, illustrating that MTKD can take advantage of both pipeline and end-to-end models.



<img src="./Figures/model.jpg" style="zoom:150%;" />



## 2. Usage

### 2.1 Requirements

- python==3.6.2
- pytorch == 1.3.1
- torchvision==0.4.2
- numpy==1.19.1
- lmdb==0.99
- PIL==7.2.0
- jieba==0.42.1
- nltk==3.5
- six==1.15.0
- natsort==7.0.1



### 2.2 Train the Model

```shell
bash ./train_model_guide.sh
```



### 2.3 Evaluate the Model

```shell
bash ./test_model_guide.sh
```



### 2.4 Datasets

We use the dataset released in [E2E_TIT_With_MT](https://github.com/EriCongMa/E2E_TIT_With_MT/tree/main).



## 3. Acknowledgement

The reference code of the provided methods are:

- [EriCongMa](https://github.com/EriCongMa)/[**E2E_TIT_With_MT**](https://github.com/EriCongMa/E2E_TIT_With_MT)
- [clovaai](https://github.com/clovaai)/**[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)**
- [OpenNMT](https://github.com/OpenNMT)/**[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)**
- [THUNLP-MT](https://github.com/THUNLP-MT)/**[THUMT](https://github.com/THUNLP-MT/THUMT)**


We thanks for all these researchers who have made their codes publicly available.



## 4. Citation

If you want to cite our paper, please use this bibtex version:

- Springer offered bib citation format

  - ```latex
    @InProceedings{10.1007/978-3-031-41676-7_28,
    author="Ma, Cong
    and Zhang, Yaping
    and Tu, Mei
    and Zhao, Yang
    and Zhou, Yu
    and Zong, Chengqing",
    editor="Fink, Gernot A.
    and Jain, Rajiv
    and Kise, Koichi
    and Zanibbi, Richard",
    title="Multi-teacher Knowledge Distillation for End-to-End Text Image Machine Translation",
    booktitle="Document Analysis and Recognition - ICDAR 2023",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="484--501",
    isbn="978-3-031-41676-7"
    }
    ```
  
- Semantic Scholar offered bib citation format

  - ```latex
    @inproceedings{Ma2023MultiteacherKD,
      title={Multi-teacher Knowledge Distillation for End-to-End Text Image Machine Translation},
      author={Cong Ma and Yaping Zhang and Mei Tu and Yang Zhao and Yu Zhou and Chengqing Zong},
      booktitle={IEEE International Conference on Document Analysis and Recognition},
      year={2023},
      url={https://api.semanticscholar.org/CorpusID:261102128}
    }
    ```
  
- DBLP offered bib citation format

  - ```latex
    @inproceedings{DBLP:conf/icdar/MaZTZZZ23,
      author       = {Cong Ma and
                      Yaping Zhang and
                      Mei Tu and
                      Yang Zhao and
                      Yu Zhou and
                      Chengqing Zong},
      editor       = {Gernot A. Fink and
                      Rajiv Jain and
                      Koichi Kise and
                      Richard Zanibbi},
      title        = {Multi-teacher Knowledge Distillation for End-to-End Text Image Machine
                      Translation},
      booktitle    = {Document Analysis and Recognition - {ICDAR} 2023 - 17th International
                      Conference, San Jos{\'{e}}, CA, USA, August 21-26, 2023, Proceedings,
                      Part {I}},
      series       = {Lecture Notes in Computer Science},
      volume       = {14187},
      pages        = {484--501},
      publisher    = {Springer},
      year         = {2023},
      url          = {https://doi.org/10.1007/978-3-031-41676-7\_28},
      doi          = {10.1007/978-3-031-41676-7\_28},
      timestamp    = {Fri, 16 Aug 2024 07:47:09 +0200},
      biburl       = {https://dblp.org/rec/conf/icdar/MaZTZZZ23.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
    ```



If you have any issues, please contact with [email](macong275262544@outlook.com).
