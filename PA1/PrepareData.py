import csv

rawdata = []

geographies = ["Spain", "France", "Germany"]

dataset_file = 'judge.csv'
cleaned_dataset_file = 'judge_clean.csv'
has_exited = False

with open(dataset_file, newline='') as csvfile:
    csvreader = csv.DictReader(csvfile, delimiter=",")
    for row in csvreader:

        clean_data = []
        clean_data.append(row["CreditScore"])

        for location in geographies:
            if row["Geography"] == location:
                clean_data.append("1")
            else:
                clean_data.append("0")

        if row["Gender"] == "Male":
            clean_data.append("1")
        else:
            clean_data.append("0")
        if row["Gender"] == "Female":
            clean_data.append("1")
        else:
            clean_data.append("0")
        clean_data.append(row["Age"])
        clean_data.append(row["Tenure"])
        clean_data.append(row["Balance"])
        clean_data.append(row["NumOfProducts"])
        clean_data.append(row["HasCrCard"])
        clean_data.append(row["IsActiveMember"])
        clean_data.append(row["EstimatedSalary"])

        if has_exited :
            clean_data.append(row["Exited"])

        if len(rawdata) == 0:
            print(clean_data)

        rawdata.append(clean_data)

print(str(len(rawdata[0])))

with open(cleaned_dataset_file, "w+") as dataset:
    titles = "CreditScore,G1,G2,G3,Male,Female,Age,Tenure,Balance,NumofProducts,HasCrCard,IsActiveMember,EstimatedSalary"
    if has_exited:
        title = titles + ",Exited\n"
    else:
        titles = titles + "\n"

    dataset.write(titles)

    for row in rawdata:
        dataset.write(row[0] + ",")
        dataset.write(str(row[1]) + ",")
        dataset.write(str(row[2]) + ",")
        dataset.write(row[3] + ",")
        dataset.write(row[4] + ",")
        dataset.write(row[5] + ",")
        dataset.write(row[6] + ",")
        dataset.write(row[7] + ",")
        dataset.write(row[8] + ",")
        dataset.write(row[9] + ",")
        dataset.write(row[10] + ",")
        dataset.write(row[11] + ",")
        dataset.write(row[12])

        if has_exited:
            dataset.write("," + row[13] + "\n")
        else:
            dataset.write("\n")
