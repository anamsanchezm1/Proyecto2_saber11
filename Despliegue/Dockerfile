FROM python:3.11-slim

WORKDIR /opt/app

COPY . .

RUN pip install --upgrade pip
RUN pip install pandas numpy joblib dash plotly flask gunicorn
RUN pip install scikit-learn==1.8.0
RUN pip install tensorflow==2.20.0 keras==3.12.0

EXPOSE 8050

CMD ["python", "despliegue.py"]
