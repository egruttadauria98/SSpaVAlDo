"""
This dictionary contains the path to the final checkpoints for the trained models.
The first key is the sample rate, the second key is the number of output sources of the model.
"""
import os

ssep_models_dict = {
  8000 : {
    2 : "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k",
    3 : "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_8k",
    4 : "",
    5 : ""
    },
  16000 : {
    2 : "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
    3 : "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k",
    4 : "",
    5 : ""
    }
}
