# 3DCV Workshop 05 - 3D Reconstruction

## Workshop Overview

In this workshop, you will use your own images of the same scene and reconstruct it in 3D using one of the recently developed geometric foundation models. You will:

1. capture or load your own set of images or a video,
2. get the camera poses and a sparse reconstruction using COLMAP,
3. identify a suitable tool or model for your data,
4. run the model and get a result,
5. visualize the resulting 3D scene or object.

This workshop will give you a first impression about the potential and challenges of recent research results.

---

## Step 1 - Capturing data

### **Capturing**

Capture RGB images of some small object or a room or scene that you want to reconstruct.

#### **Hints for capturing**

Work like a scientist. Think before you act:

- Which properties should the object have, so that the reconstruction is easy?
- Capture scenes with good lighting conditions. Consider lens distortion and reflections.
- Note for all your samples your expectations. Name and sort all captured scenes and images in a folder structure that allows you to automatically process all images using a script.

#### **Image preprocessing**

Inspect the captured images and prepare them for reconstruction. If necessary, use a tool like [SAM](https://github.com/facebookresearch/segment-anything) to segment the object from the background or create image masks.

---

## Step 2 - COLMAP

### **Reconstructing a sparse 3D model and camera poses**

**COLMAP** is the standard academic Structure-from-Motion software.

- Website: [colmap.github.io](https://colmap.github.io/)
- Documentation: [COLMAP Tutorial](https://colmap.github.io/tutorial.html)
- Source code: [GitHub colmap](https://github.com/colmap/colmap)

**WARNING:** `colmap.org` seems to be scam, avoid it.

#### **Hints for working with COLMAP**

- Run-time can be very long for a large number of images. Try to estimate first or start with a subset of your images.

## Step 3 - Choose geometric foundation model

## Preparation

Before you start, choose one geometric foundation model from this list:

[Awesome Dust3r - 3D Reconstruction](https://github.com/ruili3/awesome-dust3r#3d-reconstruction)

- check the hardware requirements and choose a model suitable to the hardware you have access to.
- check the sample results and think about a similar object or scene that you want to reconstruct.

### Models (some tested in 2025)

| Model                                                                                                                                                                        | Platforms                                                                | Year of publication                                                                                                            | Lab                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| [Dust3r](https://github.com/naver/dust3r)                                                                                                                                    | Linux, [macOS](https://github.com/naver/dust3r/issues/31), Windows (WSL) | [CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html) | Naver Labs Europe                                                                                                                 |
| [Mast3r](https://github.com/naver/mast3r) + [Mast3r-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)                                                                    | Linux, Windows (WSL)                                                     | [3DV 2025](https://arxiv.org/abs/2409.19152)                                                                                   | Naver Labs Europe                                                                                                                 |
| [Spann3r](https://github.com/HengyiWang/spann3r)                                                                                                                             | Linux, Windows                                                           | [3DV 2025](https://arxiv.org/abs/2408.16061)                                                                                   | [UCL, UK](https://hengyiwang.github.io/projects/spanner)                                                                          |
| [VGGT](https://github.com/facebookresearch/vggt)                                                                                                                             | Linux, Windows                                                           | [CVPR 2025](https://arxiv.org/abs/2503.11651) (Best Paper Award)                                                               | [Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/) and [Meta AI](https://ai.facebook.com/research/) |
| [DepthAnything3](https://github.com/ByteDance-Seed/depth-anything-3) + [DA3 Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) | Linux, Windows (not tested)                                              | [ICLR 2026](https://iclr.cc/)                                                                                                  | ByteDance                                                                                                                         |

---

## Step 4 - Get the model running

Follow the installation instructions of your chosen model. Then test the installation with the test data from the authors. If those tests are successfull, run the model with your own data and get a 3D model as result.

### **Hints for working with scientific tools**

Work like a scientist. Think before you act:

- Check the documentation (GitHub readme) of the tool you selected.
- Compare the system requirements of the tools with your computer. Usually this is not stated prominently, so you may need to check the code or issues. And be very precise about the hard- and software (driver versions, CUDA version) you have available.
- Start with the example images provided by the tool to ensure it works correctly. If not available, use synthetic images from the [NeRF dataset](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)
- Obey all warnings and error messages as they might result in later problems. Make a protocol while installing the tool.

---

## Step 5 - Final visualization

Visualize your results. Use the recommended viewer of the used tool and model and export the result as 3D model file that can be opened with MeshLab or any other tool.

---

## Submission

Submit your code as Gitlab repository.

Exclude:

- the image data you have used (too large)

Include:

- the code for the visualization,
- one screenshot of the 3D visualization,
- a 3D model file of the final result (only if smaller than 20 MB)
- a short README that explains your capture setup and which model you choose and why this one.

### Grading Criteria

- Process description (data and model choice) (25%)
- Quality of the results (40%)
- Documentation (25%)
- Experimentation and discussion of failure cases (10%)
