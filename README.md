# QuaternionTransformers

This is our Tensor2Tensor implementation of [Quaternion Transformers](https://arxiv.org/abs/1906.04393). This paper will be presented in the upcoming ACL 2019 in Florence.  

## Dependencies
1. Tensorflow 1.12.0
2. Tensor2Tensor 1.12.0
3. Python 2.7

## Usage
1. The usage of this repository follows the original Tensor2Tensor repository (e.g., t2t-datagen, t2t-trainer followed by t2t-decoder). It helps to gain familiarity on T2T before attempting to run our code.
2. Setting `--t2t_usr_dir=./QuaternionTransformers` will allow T2T to register Quaternion Transformers. To verify, using `t2t-trainer --registry_help` to verify that you are able to load Quaternion transformers.
3. You should be able to load `MODEL=quaternion_transformer` and use base or big setting as per normal.
4. Be sure to set `--hparams="self_attention_type="quaternion_dot_product""` to activate Quaternion Attention.
5. By default, Quaternion FFNs are activated for positional FFN layers. To revert and *not* use Quaternion FFNs on the position-wise FFN, set `--hparams="ffn_layer="raw_dense_relu_dense"`.

## Citation

If you find our work useful, please consider citing our paper:

```
@article{tay2019lightweight,
  title={Lightweight and Efficient Neural Natural Language Processing with Quaternion Networks},
  author={Tay, Yi and Zhang, Aston and Tuan, Luu Anh and Rao, Jinfeng and Zhang, Shuai and Wang, Shuohang and Fu, Jie and Hui, Siu Cheung},
  journal={arXiv preprint arXiv:1906.04393},
  year={2019}
}
```
