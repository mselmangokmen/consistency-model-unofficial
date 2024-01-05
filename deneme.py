
total_training_steps=50111
perc_step=total_training_steps//100
for current_training_step in range(total_training_steps):

    progress_val= (current_training_step*100)/total_training_steps
    progress_val= round(progress_val,2)

    if current_training_step%(perc_step)==0 and current_training_step!=0:
        print(progress_val)