# Airflow Lab 1 – Custom ML Pipeline with Docker & GCP VM

## Overview

This project demonstrates a **complete Apache Airflow setup** running inside **Docker** on a **Google Cloud VM**, executing a **custom machine-learning pipeline** using a real dataset.

The DAG:

1. Loads a CSV dataset
2. Preprocesses numeric features
3. Trains a KMeans clustering model
4. Saves the trained model
5. Loads the model for validation

This lab focuses on **Airflow fundamentals**, **XCom data passing**, **Dockerized execution**, and **real-world debugging** rather than production hardening.

---

## What This Lab Demonstrates

* Running **Apache Airflow 2.7.3** using Docker Compose
* Creating a **custom DAG** (not example DAGs)
* Using **PythonOperator**
* Passing data safely via **XCom (base64 + pickle)**
* Running Airflow on a **GCP Compute Engine VM**
* Debugging:

  * permissions issues
  * SQLite executor constraints
  * Docker volume pitfalls
  * Git authentication issues
* Understanding **why Airflow breaks**, not just how to fix it

---

## Project Structure

```
airflow-practice/
│
├── dags/
│   ├── airflow.py            # Main DAG definition
│   ├── data/
│   │   ├── file.csv          # Input dataset
│   │   └── test.csv
│   └── src/
│       ├── __init__.py
│       └── lab.py            # ML pipeline logic
│
├── plugins/                  # Empty (reserved)
├── logs/                     # Airflow logs (Docker-managed)
├── docker-compose.yaml
├── docker-compose.yaml.save
├── setup.sh
└── README.md
```

---

## Environment & Versions

| Component      | Version               |
| -------------- | --------------------- |
| OS             | Ubuntu 22.04 (GCP VM) |
| Docker         | Latest                |
| Docker Compose | v2                    |
| Airflow        | 2.7.3                 |
| Python         | 3.8 (Airflow image)   |
| Executor       | SequentialExecutor    |
| Metadata DB    | SQLite (dev only)     |

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/piriyajeishree410/airflow-practice.git
cd airflow-practice
```

### 2. Start Airflow

```bash
docker compose up -d
```

### 3. Verify container

```bash
docker compose ps
```

You should see:

```
airflow   apache/airflow:2.7.3   Up   0.0.0.0:8080->8080
```

### 4. Access Airflow UI

Open in browser:

```
http://<VM_EXTERNAL_IP>:8080
```

Login:

```
Username: admin
Password: admin
```

---

## DAG Details

### DAG Name

```
Airflow_Lab1_Custom
```

### Schedule

* Runs **manually** (catchup disabled)
* Can be triggered from UI

### Tasks

| Task ID                   | Purpose                            |
| ------------------------- | ---------------------------------- |
| `load_data_task`          | Reads CSV and serializes dataframe |
| `data_preprocessing_task` | Scales numeric features            |
| `build_save_model_task`   | Trains KMeans and saves model      |
| `load_model_task`         | Loads model for validation         |

All tasks execute sequentially.

---

## Data Flow (Important)

Airflow **cannot pass raw Python objects** safely between tasks.

So this project uses:

* `pickle.dumps()`
* `base64.b64encode()`

Each task:

1. Decodes input from XCom
2. Processes data
3. Re-encodes output for next task

This avoids:

* serialization crashes
* metadata DB corruption

---

## Model Storage

Models are saved to:

```
/opt/airflow/working_data/model/
```

Why?

* `/opt/airflow/dags` is **read-only at runtime**
* Writing inside DAG folders causes permission errors
* This mirrors real Airflow best practices

---

## Key Issues Faced (And Why)

### SQLite + LocalExecutor crash

**Reason:** SQLite does not support LocalExecutor
**Fix:** Use `SequentialExecutor`

---

###  Airflow logs permission errors

**Reason:** Mounting `./logs` from host causes UID/GID mismatch
**Fix:** Let Docker manage logs internally

---

###  Cannot write to `dags/src`

**Reason:** DAG code is mounted read-only by design
**Fix:**

* Modify files **locally**
* Commit & pull
* Restart container

---

### Git push fails with password

**Reason:** GitHub no longer supports password auth
**Fix:** Use **Personal Access Token (PAT)**

Required permissions:

```
Repository → Contents → Read & Write
```

---

## Expected Output

* Airflow UI shows:

  * 1 DAG
  * All tasks green
* Logs show:

  ```
  User "admin" created with role "Admin"
  Listening at: http://0.0.0.0:8080
  ```
* Model file saved in:

  ```
  working_data/model/
  ```

---

## Warnings You’ll See (Expected)

These are **normal for a lab**:

* “Do not use SQLite in production”
* “Do not use SequentialExecutor in production”

They are **educational**, not errors.


---

## Final Status

- Airflow UI accessible
- DAG loads without import errors
- Tasks execute successfully
- ML pipeline runs end-to-end
- Git repo clean and reproducible

---



