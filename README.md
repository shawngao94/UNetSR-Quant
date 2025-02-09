# UNetSR-Quant
An example of quantization-aware training (QAT) applied to a UNet-based super-resolution approach

Run main training for QAT or float32 model training.
```
python main.py
```

Evaluation of the size and performance of trained QAT.
```
python quant_test.py
```

modified from  
https://github.com/htqin/QuantSR  
https://github.com/Mnster00/simplifiedUnetSR

references  
QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution
