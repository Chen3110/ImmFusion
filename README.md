# ImmFusion: Robust mmWave-RGB Fusion for 3D Human Body Reconstruction in All Weather Conditions

## Implementation Details for DeepFusion and TokenFusion
### Framework
![framework](imgs\framework.png)

### DeepFusion

DeepFusion adopts a Transformer-based module to fuse image and point features. Specifically, it transforms the point features to the queries and image features to the keys/values and then aggregates image features to the point features. As DeepFusion is a generic Transformer-based block that is incompatible with the dimension-reduction mechanism, we adopt the parametric reconstruction pipeline by replacing the detection framework of DeepFusion with linear projection to regress SMPL-X parameters. To be specific, we utilize the image backbone ResNet (same as original DeepFusion) and the point backbone PointNet++ to extract image and point local features from images and radar, respectively. Then, we adopt MLPs to transform local point features and image features to queries and keys/values, respectively. Then we feed the features into a general Transformer module to fuse the information. The fused features are then concatenated with the original point features. At last, we adopt the max pooling and MLPs to regress features to the SMPL-X parameters.

### TokenFusion

For the TokenFusion method, we implement it without structure altering. Specifically, as images and point clouds are heterogeneous modalities, we train two standalone Transformers for the two modalities, respectively. We plug Score Nets among the Transformer layers to dynamically predict the importance of local features obtained by the backbones (same as that used in DeepFusion). The Score Net is implemented using an MLP to dynamically score the feature tokens. The joint/vertex tokens with scores lower than the threshold are substituted with corresponding ones from the other modality. At last, the pruned joint/vertex features of two modalities are integrated to regress final coordinates by MLPs.

Code will be available soon.
