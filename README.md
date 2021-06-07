# qna-recommender

## How to run
1. Install all the requirements
```
pip install -r requirements.txt
```
2. Get the cleaned dataset from here
```
https://drive.google.com/drive/folders/1pXKxNbhn4-O9iiE9Qv4N1P8tUeaS0F4U?usp=sharing
```
3. Put dataset into folder data/raw
4. Run following command to train the model
```
python app.py -ctr
python app.py -lstr
python app.py -clstr
```
5. After training, change model_path on app.py (line 58), and run this command to get the recommendation based on question Id
```
python app.py -r
```

## Pretrained Model
```
https://drive.google.com/drive/folders/1D3eXg2ALU94z7ysw7OaRIt9-UlqHrgoS?usp=sharing
```