# omnidirectional-viewer-sample-using-opencv
OpenCVを利用した360度画像の簡易ビューアです。

https://user-images.githubusercontent.com/37477845/181903194-a6ce5e2f-faa5-47e8-a0e7-ab705e1bbec3.mp4

# Requirement 
* opencv-python 4.5.3.56 or later

# Demo
デモは2種類あります。<br>
1. コマンドライン操作で画像を変換し保存するデモ（01_simple_image_convert.py）
2. マウス操作でピッチ・ヨー・ズームの視点を操作するデモ（02_omnidirectional_viewer.py）

### 1. コマンドライン操作で画像を変換し保存するデモ（01_simple_image_convert.py）
```bash
python 01_simple_image_convert.py
```
<details>
<summary>パラメータ</summary>
  
* --roll<br>
ロール角の指定<br>
デフォルト：0
* --pitch<br>
ピッチ角の指定<br>
デフォルト：0
* --yaw<br>
ヨー角の指定<br>
デフォルト：0
* --viewpoint<br>
半径1の球体に正規化した際のX軸視点位置<br>
デフォルト：-1.0
* --imagepoint<br>
半径1の球体に正規化した際のX軸投影位置<br>
デフォルト：1.0
* --sensor_size<br>
半径1の球体に正規化した際の投影幅<br>
デフォルト：0.561
* --width<br>
出力画像の横幅<br>
デフォルト：640
* --width<br>
出力画像の縦幅<br>
デフォルト：360
* --image<br>
入力画像パス<br>
デフォルト：sample.png
* --output<br>
出力画像パス<br>
デフォルト：output.png
</details>

### 2. マウス操作でピッチ・ヨー・ズームの視点を操作するデモ（02_omnidirectional_viewer.py）
ウィンドウ上でマウス左ドラッグでピッチ・ヨー操作、ホイールでズーム操作が出来ます。<br>
※ロール操作には対応していません
```bash
python 02_omnidirectional_viewer.py
```
<details>
<summary>パラメータ</summary>
  
* --viewpoint<br>
半径1の球体に正規化した際のX軸視点の初期位置<br>
デフォルト：-1.0
* --imagepoint<br>
半径1の球体に正規化した際のX軸投影の初期位置<br>
デフォルト：1.0
* --sensor_size<br>
半径1の球体に正規化した際の投影幅<br>
デフォルト：0.561
* --width<br>
出力画像の横幅<br>
デフォルト：640
* --width<br>
出力画像の縦幅<br>
デフォルト：360
* --image<br>
入力画像パス<br>
デフォルト：sample.png
* --movie<br>
入力動画パス ※指定時はimageオプションより優先<br>
デフォルト：指定なし
</details>

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
omnidirectional-viewer-sample-using-opencv is under [MIT License](LICENSE).

また、水中の360度動画は、[360あーる・てぃー・しー](https://360rtc.com)様の<br>
「[浅瀬の水中を色鮮やかな魚が泳ぐ沖縄県恩納村恩納 No.1](https://360rtc.com/videos/apogama001/)」を利用しています。
