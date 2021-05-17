# NeRF-Pytorch Code Reading

This repository is forked from [[yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)] and aims to record the code-reading.

## Overview

The main training code is in the `run_nerf.py`, and `run_nerf_helpers.py` contains lots of helper function including the NeRF model structure, the positional embedding structure, as well as how to get_rays, how to get ndc rays, how to perform hierarchical sampling (in section 5.2 of ECCV'20 paper)

The `run_nerf.py` mainly contains a `train()` function, which is the main training process of a NeRF model. Specifically, it contains `Data preparation`, `Model and Optimizer construction`, and `training loop`. These three parts are also the main steps to start training a pytorch model.

- **Args Parser**: obtain settings of training a NeRF model.
- **Data preparation**: create the dataload for different type of dataset. It seems the orginal RGB(a) images will be loaded into the memory. Since a NeRF model can only trained on images collected for ONE scene once, in practice, the number of images would be small. It thus all the images for training can be loaded. The dataloader will return 
  - Image array, 
  - Camera pose (angle) for each image, 
  - Rendered pose (camera to world transformation matrices),
  - Height, width of the image, focal length of that image
  - i_split: TODO: 'UNCLEAR' PART, for split dataset for different type of dataset, maybe?
- **Model & Optimizer construction**: Declare the model and optimizer in the `create_nerf()` function. This function will return:
  - render_kwargs_train: A dictionary contains the paramters
    ```python
    render_kwargs_train = {
        "network_query_fn": network_query_fn,    # main body of forward network, create by using a lambda function with run_network(), which contains the positional embedder forward net of 3D corrdinates and view direction, and batchify() function of NeRF network forward pass.  
        "perturb": args.perturb,                 # perturb controls the sampling interval in a camera ray
        "N_importance": args.N_importance,       # TODO: UNCLEAR: Importance sampling?
        "network_fine": model_fine,              # TODO: UNCLEAR: Refined version of Nerf Model?
        "N_samples": args.N_samples,             # Number of sampling points
        "network_fn": model,                     # NeRF model
        "use_viewdirs": args.use_viewdirs,       # w/ or w/o using view directions (& positional embedding)
        "white_bkgd": args.white_bkgd,           # whether or not whiten the backgound color (accroding to the alpha channel of the image) 
        "raw_noise_std": args.raw_noise_std,     # TODO: UNCLEAR: Random noise for input image?
    }
    ```
  - render_kwargs_test: Same with render_kwargs_train, but set `perturb=False` and `raw_noise_std=0.0`.
  - start: starting global step
  - grad_vars: varible set, which contains the parameters need to be updated by computing gradient.
  - optimizer: optimizer, paramter and learning rate updating policy.
- **Training loop**: Main loop to training NeRF model with batch ray data and update the model parameters in each iteration. It mainly contains the following steps:
  - `get_rays()`: Prepare training data for this iteration, get rays dataset for training, 
  - `render()`: Model prediction. output the prediction for each ray, prediction the RGB values for each ray.
  - `zero_grad the optimizers`: optimizers.zero_grad(), sets the gradients of all optimized parameters to zero.
  - `Calcluating loss: for backward and logging`
  - `Calcluating PSNR metric: for logging`
  - `loss.backward()`: calcuating gradient for paramsters
  - `optimizer.step()`: updating the parameters
  - `Update learning rate:`
  - `if model should be saved, then save model`
  - `if model should be validated/tested, then test model`
  - `if log should be printed, then print log`
  - `update the global iteration step`.
  - `continue the training loop until the end.`

##  Functions and operations in Training loop
The forward pass of the model is complex. There are many details in those functions like `get_rays()` and `render()`. Thus, here, I read this part of code and record some comments.
- **get_rays()**: Data preparation, there are two mode to get training rays, the first one is get rays by sampling from different images, the second one is getting rays from one image, and the image is randomly selected. 
  - `get_rays()` function is defined in the `run_nerf_helper.py` file. 
    - The function will return the camera rays shoting from each pixel locations with the start point (ray_o) and end point (rays_d) in the 3D space. 
    - The screen space coordiated will be firsting translated to the center of the screen, and transformed by using the camara-to-world(c2w) transformation matrices and the camera intrinsic (focal length). 
    - The start point is the translation parameters in the c2w matrices.
    - The function will return the `rays_o` and `rays_d` with size (H, W, 3)
  - `render()` funtion defines the main forward pass of the model
    - it need to process the ray data by concatenating the `near` and `far` parameters, as well as whether or not using view directions. thus the input ray datas now contains `rays_o`, `rays_d`, `near`, `far`, `view direction`, a total of five kinds of data, with feature length `3+3+1+1+3=11`. P.S. the view direction is actually the normalized direction with respect to `rays_d`.
    - `batchify_rays()` function will perform the `render_rays()` process for each data chunk, and gather all the results for all the input data chunks. 
    - `render_rays()`: this function contains
      - Parsing the input ray data into five parts, i.e., `rays_o`, `rays_d`, `near`, `far`, `view direction`.
      - Sampling the 3D points on the ray defined by the `rays_o` and `rays_d` with ray marching distance according to the space boundary defined by the `near` and `far` parameters.
      - `network_query_fn()` get the raw output for each 3D points on each camera ray.
      - `raw2outputs()` accumulating the raw outputs on each camera ray, and predict alpha, and RGB value for the camera ray.
      - `if N_importance > 0`, then perform importance sampling, and also the `network_query_fn()` and `raw2outputs()` forward pass by using the importance sampling points.
      - return a dictionary with `{"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}`

## Functions and operations to forward networks
The `network_query_fn()` and `raw2outputs()` functions denote the mainly forward pass of the whole model. Specifically,
- `network_query_fn()` is a lambda function defined by the `run_network()` function.
- The input of this `run_network()` is
  - `input`: input 3D points sampled on the rays as mentioned in the `render_rays()` function; size of the `Torch.Tensor` should be `[N_rays, N_samples, 3]`;
  - `viewdir`: view direction, size of the `Torch.Tensor` should be `[N_rays, 3]`;
  - `network_fn`: NeRF model, i.e., MLP structure, main body of the NeRF network.
- The whole process of the `run_network()` can be divided into follow steps:
  - `Positional embedding for 3D points`: `input` will be firstly flatten to `[N_rays*N_samples, 3]`. For each point, peform positional embedding. 
  - `Positional embedding for view direction (optional)`:expand the view direction with the shape of the input 3D points, perform positional embedding, and then concatenate the view direction embedding with the point positional embedding. 
  - `batchify()` function uses the positional embedding as input
    - Perform network forward by feeding chunk of 3D point embeddings.
    - Reshape to `[N_rays, N_samples, 4]`, 4 denotes `[rgb, alpha]` for each point.
  - Output the result.
- `raw2outputs()` function transforms the raw RGB and alpha values of sampled 3D points on a camera ray to a RGB and alpha values.
- The input of the `raw2outputs()` function is:
  - `raw`: the predicted rgb and alpha values for the sampled 3D points along each camera ray, the size of raw tensor `[N_rays, N_samples, 4]`.
  - `z_val`: the accumulated ray marching distance for each sample points within `near` and `far` bound.
  - `rays_d`: view direction of each ray.
- Forward passing process of the `raw2outputs()` function.
  - Get the bin distance on the ray marching process. (TODO: Padding the last distance to a large value, which denotes infinite?)
  - Get the norm of the ray direction
  - `dists`, Distance of each bin on each ray are re-weighted by the norm of each ray direction, return `dists`.
  - apply `sigmoid` function on the first three values in the raw output to get the rgb values (range from 0~1, each value are independent)
  - apply a `raw2alpha()` function on the fourth value (alpha prediction + noise value) and `dists` value on each sampled point. the `raw2alpha` is a lambda function which is defined by `1.0 - exp(-relu(raw) * dists)`. It can be observed, if the distance is infinite, the alpha values should be 1.0 accroding to the equation of this function.
  - According the Equation (5) in the ECCV'20 paper of NeRF, the next step aims to accumuate the `1.0-alpha` values to the previous sampling position.
    - here, `torch.cumprod` operation will return the cumulative production of elements by following equations:
    - y_1 = x_1
    - y_2 = x_1 * x_2
    - y_i = x_1 * x_2 * ... * x_i
    - y_Nsample = x_1 * x_2 * ... * x_nsample
  - Then the alpha will multipled by the accumuated `1-alpha` values to get the `weights` as discribed in the second part of Equation (5)
  - re-weight the `rgb` values in the each sampling points, and aggreated them to a single set of RGB values for each camera ray. (See the first part of the Equation (5).)
  - re-weight the accumulated ray marching distance `z_val` and sum the values together to get a depth value for each camera ray.
  - return `rgb_map, disp_map, acc_map, weights, depth_map`.
  - end of the `render_rays()` function
  - back to the `batchify_rays()` function
  - back to the `render()` function
  - back to the `train()` function


## Summary
The relationship between code execution and functions can be represented by the following hierarchy.

```shell
if __name__ == '__main__':
├── train()
|   ├── config_parser()
|   ├── parser.parse_args()
|   ├── load_data()
|   ├── create_nerf()
|   |   ├── get_embedder()
|   |   ├── NeRF()
|   |   ├── grad_vars = list(model.parameters())
|   |   ├── network_query_fn()
|   |   ├── optimizer()
|   |   ├── load_ckpt() (optional)
|   └── train_loop()
|       ├── prepare training rays for i-th iteration
|       ├── render() forward pass to get prediction
|       |   ├── get_rays()
|       |   ├── ndc_rays() (if needed)
|       |   ├── get near, far bound, get view direction (if needed)
|       |   ├── batchify_rays()
|       |   |   ├── for each data chunk 
|       |   |   |   ├── render_rays()
|       |   |   |   |   ├── parsing input data with [`ro`, `rd`, `n`, `f`, `vd`]
|       |   |   |   |   ├── z_vals by spliting N_sample bins in 0.0, 1.0 in the near and far boundary.
|       |   |   |   |   ├── get samping points between `ro` and `rd`
|       |   |   |   |   ├── network_query_fn()
|       |   |   |   |   |   ├── run_network()
|       |   |   |   |   |   |   ├── embed_fn()
|       |   |   |   |   |   |   ├── embeddirs_fn(if needed)
|       |   |   |   |   |   |   ├── batchify()
|       |   |   |   |   |   |   |   ├── model(NeRF).forward()
|       |   |   |   |   |   |   |   └── gather all the prediction
|       |   |   |   |   |   |   └── return outputs
|       |   |   |   |   ├── raw2outputs()
|       |   |   |   |   |   ├── rgb = torch.sigmoid(raw[..., :3])
|       |   |   |   |   |   ├── alpha = raw2alpha()
|       |   |   |   |   |   ├── weights = alpha * torch.cumprod(dists)
|       |   |   |   |   |   ├── rgb_map = torch.sum(weights[..., None] * rgb)
|       |   |   |   |   |   ├── depth_map = torch.sum(weights * z_vals, -1)
|       |   |   |   |   |   ├── acc_map = torch.sum(weights, -1) 
|       |   |   |   |   |   └── return outputs
|       |   |   |   |   ├── if N_importance > 0
|       |   |   |   |   |   ├── network_query_fn(network_fine)
|       |   |   |   |   |   ├── raw2outputs()
|       |   |   |   |   └── return rgb_map, dispmap, accmap
|       |   |   └── gather all thre results
|       ├── optimizer.zero_grad()
|       ├── calculate loss
|       ├── calculate psnr
|       ├── calcuate extra loss (sideway outputs)
|       ├── calcuate extra psnr (sideway outputs)
|       ├── loss.backward()
|       ├── optimizer.step()
|       ├── update learning rate
|       ├── save ckpt (if needed)
|       ├── save mp4 (test resutls if needed)
|       ├── valiate on test/val set (if needed) 
|       ├── print logs (if needed)
|       └── global iteration number update
└── end of training / validating
```