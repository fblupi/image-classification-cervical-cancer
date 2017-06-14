| Número de solución | Descripción | Algoritmo y software empleado | Resultado (test) | Resultado (training) | Posición | Fecha y hora |
| ---:| --- | --- | ---:| ---:| ---:|:---:|
| 1 | CNN simple learning from scratch con datos de train básicos | R, mxnet | 1.11456 | 1.0688049 | 671 | 5/6/2017 10:25 | 
| 2 | CNN simple learning from scratch con más datos de train y balanceados | R, mxnet | 1.06771 | 1.033483 | 646 | 7/6/2017 11:17 |
| 3 | CNN simple learning from scratch con más datos de train (no balanceados) a 128x128 durante 10 épocas | Python, Keras | 0.9991 | 0.95566 | 412 | 8/6/2017 13:23 |
| 4 | CNN simple learning from scratch con más datos de train (no balanceados) pero a 32x32 durante 20 épocas | Python, Keras | 0.91442 | 0.9575 | 344 | 8/6/2017 17:32 |
| 5 | CNN simple learning from scratch con más datos de train (no balanceados) a 128x128 durante 50 épocas | Python, Keras | 0.94097 | 0.8224 | 347 | 8/6/2017 18:35 |
| 6 | CNN simple learning from scratch con más datos de train (no balanceados) a 128x128 durante 20 épocas | Python, Keras | 0.91071 | 0.8804 | 342 | 8/6/2017 19:16 |
| 7 | CNN bastante más compleja learning from scratch con más datos de train (no balanceados) a 128x128 durante 15 épocas con un batch size muy grande | Python, Keras | 0.99942 | 1.0444 | 344 | 8/6/2017 22:50 |
| 8 | CNN simple learning from scratch con más datos de train (filtrados y algo más balanceados) a 128x128 durante 10 épocas con un batch size muy pequeño | Python, Keras | 0.87668 | 0.8261 | 282 | 9/6/2017 11:23 |
| 9 | Primer intento con finetuning en R | R, mxnet | 2.24609 | 1.2871 | 282 | 9/6/2017 11:34 |
| 10 | Fine tuning ResNet50 con capa FC de tamaño 1024. Epochs=5, batch size=15 | Python, Keras | 0.91451 | 0.6656 | 288 | 10/6/2017 16:00 |
| 11 | Fine tuning ResNet50 con capa FC de tamaño 512. Epochs=10, batch size=15 | Python, Keras | 0.84681 | 0.7661 | 250 | 10/6/2017 22:00 |
| 12 | Fine tuning ResNet50. Epochs=25, batch size=50, sample per epoch=2000 | Python, Keras | 1.11204 | 0.8318 | 251 | 11/6/2017 9:03 |
| 13 | Fine tuning ResNet50 con capa FC de tamaño 512. Epochs=5, batch size=15 | Python, Keras | 0.89159 | 0.8275 | 251 | 11/6/2017 15:15 |
| 14 | Fine tuning ResNet50 con capa FC de tamaño 512. Epochs=10, batch size=15 | Python, Keras | 0.90005 | 0.7875 | 251 | 12/6/2017 15:15 |
| 15 | CNN simple learning from scratch con más datos de train balanceados a 128x128 durante 20 épocas con un batch size muy pequeño | Python, Keras | 0.99534 | 0.7298 | 255 | 12/6/2017 18:13 |
| 16 | Fine tuning ResNet50. Epochs=50, batch size=32, sample per epoch=2000 | Python, Keras | 0.91639 | 0.7352 | 259 | 13/6/2017 21:50 |
| 17 | Fine tuning ResNet50. Epochs=75, batch size=32, sample per epoch=2000 | Python, Keras | 0.90753 | 0.7513 | 259 | 13/6/2017 21:51 |
| 18 | Fine tuning ResNet50. Epochs=100, batch size=32, sample per epoch=2000 | Python, Keras | 0.95160 | 0.7130 | 259 | 13/6/2017 21:52 |
| 19 | Fine tuning ResNet50. Epochs=125, batch size=32, sample per epoch=2000 | Python, Keras | 0.96648 | 0.6980 | 259 | 13/6/2017 21:53 |
| 20 | Fine tuning ResNet50. Epochs=75, batch size=48, sample per epoch=2000 | Python, Keras | 0.9170 | 0.7135 | 259 | 13/6/2017 22:11 |
| 21 | Fine tuning ResNet50con capa FC de tamaño 512. Epochs=75, batch size=48 | Python, Keras | 0.85543 | 0.3170 | 259 | 14/6/2017 11:20 |
| 22 | Fine tuning ResNet50con capa FC de tamaño 512. Epochs=70, batch size=48 | Python, Keras | 0.85133 | 0.3699 | 259 | 14/6/2017 11:25 |
| 23 | Fine tuning ResNet50con capa FC de tamaño 512. Epochs=50, batch size=48 | Python, Keras | 0.82368 | 0.5239 | 229 | 14/6/2017 11:30 |
| 24 | Fine tuning ResNet50con capa FC de tamaño 512. Epochs=60, batch size=48 | Python, Keras | 0.80378 | 0.4498 | 195 | 14/6/2017 11:35 |
| 25 | Fine tuning ResNet50con capa FC de tamaño 512. Epochs=65, batch size=48 | Python, Keras | 0.83104 | 0.4193 | 195 | 14/6/2017 11:40 |