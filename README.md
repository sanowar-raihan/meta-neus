## MetaNeuS: Meta-Learning for Neural Implicit Surface of Object Categories
**MetaNeuS** uses Meta-Learning to learn a template shape of an object category from a database of multiview images. This category template is encoded in the weights of the network as a signed distance function(SDF). Starting from this meta-learned template, we can quickly reconstruct a novel object at test time using a small number of views.  

This project builds on the NeurIPS'21 paper ["NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction"](https://arxiv.org/abs/2106.10689). NeuS is a scene-specific method that requires a large number of multiview inputs. This work extends NeuS using meta-learning to handle unseen objects at test time, and also enables sparse view 3d reconstruction.  

https://user-images.githubusercontent.com/71722137/147431106-428c9da0-79d1-48eb-a591-5a5b706cc7f5.mp4

Initializing with meta-learned weights also allows other applications like:
* [Geometry Interpolation](#geometry-interpolation)
* [Appearance Interpolation](#appearance-interpolation)

## Geometry Interpolation

[NeuS](https://arxiv.org/abs/2106.10689) already disentangles the geometry and appearance of an object into two separate networks. However, when the optimization starts from a standard initialization, weight space interpolation doesn't produce any meaningful result. But when it starts from a meta-learned initialization, we can interpolate the object geometry by interpolating the weights of the SDF network (while keeping the appearance constant).

https://user-images.githubusercontent.com/71722137/147683784-522a0098-7792-4744-9935-02713ed227ab.mp4

## Appearance Interpolation

Similarly, with meta-initialized networks, we can interpolate the object appearance by interpolating the weights of the color network while keeping the geometry fixed.

https://user-images.githubusercontent.com/71722137/147683876-db157da8-ae9e-48ac-9a8f-7190cdb812ff.mp4

## Usage

#### Environment
* Python 3.8
* PyTorch 1.9
* NumPy, PyMCubes, imageio, imageio-ffmpeg

#### Data
Download the dataset from this [drive link](https://drive.google.com/drive/folders/1GSHNHkNfcivGRZZS3nSOCEZcZeHJq6hk?usp=sharing). This is a modified version of the [learnit dataset](https://www.matthewtancik.com/learnit); I have normalized the scenes such that the objects are located inside a unit sphere.

#### Train
Train the NeuS model on a particular ShapeNet class with meta-learning:  
```shell
python train.py --config ./configs/$class.json
```

#### Evaluate
Optimize the meta-trained model on sparse views of unseen objects and report test results on held-out views:
```shell
python test.py --config ./configs/$class.json --meta-weight META_WEIGHT_PATH
```
It also saves the scene weights, the extracted 3D mesh and a 360-degree video for each test object in the `./results` directory.

#### Interpolation
Interpolate the geometry or appearance of two test objects:
```shell
python interpolate.py --config ./configs/$class.json --first-weight FIRST_PATH --second-weight SECOND_PATH --property PROPERTY
```
It will generate an interpolation video in the `./results` directory. Here `FIRST_PATH` / `SECOND_PATH` is the path to the weight file for any two test objects. PROPERTY is the property to interpolate, either `geometry` or `appearance`.

## Acknowledgments

I have used the following repositories as a reference for this implementation:

* [NeuS](https://github.com/Totoro97/NeuS)
* [IDR](https://github.com/lioryariv/idr)
* [learnit](https://github.com/tancik/learnit)
* [learn2learn](https://github.com/learnables/learn2learn)  

Thanks to the authors for releasing their code!
