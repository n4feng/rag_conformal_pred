# Test
Expirements are mainly done in /rag/test.ipynb and ./conformal.ipynb notebook, will consider write test seperately for each different data source later

data analyzing script are stored in /data/mining.ipynb notebooks. It contains some refactor ploting method that can easily do analyzing and plot required figures per dataset and method
support options: "factscore_gpt", "factscore_similarity", ("hotpotqa", "popqa", "medication" +"_similarity")
only similarity method support conditional conformal prediction
make sure hyper-parameter **a** matches in all metadata file names. If not, you can use scripts in test.ipynb, conformal.ipynb and mining.ipynb to generate related metadata files


# Data
Contain following datasource: factscore, popQA, hotpotQA, medicationQA
popQA Data Source: https://huggingface.co/datasets/akariasai/PopQA
Wiki Data Dump (enwiki-20230401.db): https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=drive_link

More introduction is updated in overleaf https://www.overleaf.com/project/678e8c57be330bdf525d14da
Check main.tex for explaination
