def return info(customer_id):
  gender = ""
  Age = ""
  if df.loc[customer_id, "CODE_GENDER_M"] == 1:
    gender = "Male"
  else:
    gender = "Female"
