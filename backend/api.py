import os
import tempfile
import torch
import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Import your model and preprocessor classes
from main import create_efficient_model, BRaTSPreprocessor








