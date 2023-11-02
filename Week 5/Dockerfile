FROM python:3.8-slim
WORKDIR /DataGlacier/Week5
COPY Scripts/ /DataGlacier/Week5/Scripts/
COPY model.pkl /DataGlacier/Week5/
COPY requirements.txt /DataGlacier/Week5/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "Scripts/app.py"]
