FROM python:3.10-slim

WORKDIR /opt/app

COPY . .

RUN pip install --upgrade pip
RUN pip install pandas numpy scikit-learn joblib dash plotly flask gunicorn tensorflow

EXPOSE 8050

CMD ["python", "despliegue.py"]
