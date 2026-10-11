# Third-party notices

The repository's original software is licensed under the [MIT License](LICENSE). Third-party code and assets retain their respective copyright notices and license terms. The notices below document the upstream sources identified in the implementation; they are not an exhaustive license inventory of installed dependencies.

## Grad-CAM inspiration: pytorch-grad-cam

[XAI/GradCAM.py](XAI/GradCAM.py) identifies [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) as an inspiration. Its upstream copyright notice and [MIT license text](https://github.com/jacobgil/pytorch-grad-cam/blob/master/LICENSE) are reproduced below for attribution.

```text
MIT License

Copyright (c) 2021 Jacob Gildenblat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Calibration adaptation: scikit-learn

The `calibration_curve` function in [calibration/metrics.py](calibration/metrics.py) identifies an adaptation of scikit-learn. The following copyright notice and [BSD 3-Clause license text](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/COPYING) come from scikit-learn 1.6.1, the version recorded in this repository's dependency manifests. These terms continue to apply to the adapted third-party portions.

```text
BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## InsightFace code and pretrained models

[face_detection.py](face_detection.py) uses InsightFace. According to the [upstream license policy](https://github.com/deepinsight/insightface#license), InsightFace code is MIT-licensed, while its supplied training data and pretrained models are available for non-commercial research, subject to any separately obtained license. This includes models downloaded automatically by the Python library.

The MIT license for this repository does not grant commercial-use rights to those pretrained models. Obtain any required model permissions from their owners before reuse.

## Datasets and other external assets

Permissioned UNIFESP, iCOPE and iCOPEvid datasets, patient images and videos, identifying metadata, perception heatmaps and third-party pretrained weights are not licensed by this repository's MIT license. Access, use and redistribution remain subject to the respective owners' terms and permissions. See the [dataset guidance](README.md#datasets-and-preparation) and obtain pretrained-model terms from the asset providers.
