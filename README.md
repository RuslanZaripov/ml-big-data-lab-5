# ml-big-data-lab-5

## Description

Stack: Cassandra, PySpark

## PySpark configuration

- execute the following command to know which host port has been mapped to the container's port 8888

```bash
docker port pyspark-notebook 8888
```

- fetch the notebook token. Your output should resemble this URL: `http://127.0.0.1:8888/lab?token=YOUR_TOKEN_HERE`

```bash
docker logs --tail 3 pyspark-notebook
```

### Steps

- Start docker containers

```bash
docker compose up --build
```

- Run clusterization algorithm

```bash
docker exec spark python src/clusterize.py
```

### Project Structure

```text
.
├── README.md
├── conf
│   └── spark.ini
├── docker-compose.yml
├── logfile.log
├── notebooks
│   ├── openfoodfacts_clustering.ipynb
│   ├── openfoodfacts_preprocessing.ipynb
│   └── word_count.ipynb
├── requirements.txt
├── sparkdata
│   ├── en.openfoodfacts.org.products.csv
│   ├── googleplaystore_user_reviews.csv
│   └── predictions.csv
├── src
│   ├── clusterize.py
│   └── logger.py
└── static
    └── Лабораторная работа 5 (весна 2025).pdf
```
