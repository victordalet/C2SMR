# C2SMR - INSTALLATION TEST DETECTOR 

---



## Already exist
### Generate record file

```shell
py generate_tfrecord.py --csv_input=C:\Users\dalet\C2SMR\test\data\noyade\train\_annotations.csv --image_dir=C:\Users\dalet\C2SMR\test\data\noyade\train --output_path=train.record
```
```shell
py generate_tfrecord.py --csv_input=C:\Users\dalet\C2SMR\test\data\noyade\test\_annotations.csv --image_dir=C:\Users\dalet\C2SMR\test\data\noyade\test --output_path=test.record
```

---


## Train models with Tensorflow 2

### Use this collab url

https://colab.research.google.com/drive/1lorhD7iKZFx2LIcfPbkGVYm_3axotUKj?usp=sharing
- give record file and labelmap.pbtxt
- get the saved_model folder


---

## Test result

````shell
py detection_picture.py
````