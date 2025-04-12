# SMVDR

Code for paper: Like an Ophthalmologist: Dynamic Selection Driven Multi-View Learning for Diabetic Retinopathy Grading

You can train the model using `python train.py` and test it with `python test.py`. You can train the model using `python train.py` and test it with `python test.py`.  Pretrained weights can be downloaded from [this link](https://drive.google.com/file/d/1pschFLlmKX0HODRJuKAMLAa9-69xF0I1/view?usp=drive_link).

For **four-view inference**, each of the four images must be pre-processed using the provided script `process_image.py`,  
and lesion information should be obtained via [HACDR-Net](https://github.com/xqh180110910537/HACDR-Net).



```bibtex
@inproceedings{luo2025,
  title={Like an Ophthalmologist: Dynamic Selection Driven Multi-View Learning for Diabetic Retinopathy Grading},
  author={Luo, Xiaoling and Xu, Qihao and Wu, Huisi and Liu, Chengliang and Lai, Zhihui and Shen, Linlin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={19224-19232},
  year={2025}
}

