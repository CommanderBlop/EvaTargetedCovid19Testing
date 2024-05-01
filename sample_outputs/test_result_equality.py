import csv

def compare_csv_files(file1, file2):
    with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
        reader1 = csv.reader(csv_file1)
        reader2 = csv.reader(csv_file2)
        
        next(reader1)
        next(reader2)
        # for row1, row2 in zip(reader1, reader2):
        #     # Ignore the difference in boolean values
        #     row1 = [value.lower() if isinstance(value, str) else value for value in row1]
        #     row2 = [value.lower() if isinstance(value, str) else value for value in row2]

        #     # Convert values to appropriate types before comparing
        #     row1[0] = str(row1[0])
        #     row1[1:5] = [round(float(value), 1) for value in row1[1:5]]
        #     row2[0] = str(row2[0])
        #     row2[1:5] = [round(float(value), 1) for value in row2[1:5]]

        #     # Compare the rows
        #     if row1 != row2:
        #         print(row1, row2)
        #         return False
    # return True
        count = 0
        for row1, row2 in zip(reader1, reader2):
            # Ignore the difference in boolean values
            row1 = [value.lower() if isinstance(value, str) else value for value in row1]
            row2 = [value.lower() if isinstance(value, str) else value for value in row2]
            # Convert values to appropriate types before comparing
            row1[0] = str(row1[0])
            row1[1:2] = [str(value) for value in row1[1:2]]
            row2[0] = str(row2[0])
            row2[1:2] = [str(value) for value in row2[1:2]]
            # Compare the rows
            if row1 != row2:
                count += 1
        
        return count

# Usage example
file1 = 'test_results_2020-09-01.csv'
file2 = 'r_test_results_2020-09-01.csv'
# file1 = 'country_gittins_2024-04-29.csv'
# file2 = 'r_country_gittins_2024-04-29.csv'

# if compare_csv_files(file1, file2):
#     print("Both CSV files have the same attributes.")
# else:
#     print("Attributes in the CSV files are not the same.")
print(compare_csv_files(file1, file2))