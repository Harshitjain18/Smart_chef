class KitchenTask:
    def __init__(self, task_id, arrival_time, prep_time, cook_time, attention_time):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.prep_time = prep_time
        self.remaining_prep_time = prep_time  # Remaining preparation time
        self.cook_time = cook_time
        self.remaining_cook_time = cook_time  # Remaining cooking time
        self.attention_time = attention_time
        self.remaining_attention_time = attention_time  # Remaining attention time
        self.priority = 0  # Baseline priority

    def __str__(self):
        return f"Task {self.task_id}: Arrival Time={self.arrival_time}, Prep Time={self.prep_time}, Cook Time={self.cook_time}, Attention Time={self.attention_time}, Priority={self.priority}"


def kitchen_task_scheduling(tasks):
    time_elapsed = 0
    completed_tasks = []
    current_task = None

    while tasks or current_task:
        if not current_task and tasks:
            # Get the task with the highest priority
            tasks.sort(key=lambda x: (x.priority, x.arrival_time))
            current_task = tasks.pop(0)
            # Simulate food preparation time
            time_elapsed += current_task.remaining_prep_time

        # Check if there are higher priority tasks arrived
        if tasks and tasks[0].arrival_time <= time_elapsed:
            highest_priority_task = tasks[0]
            if highest_priority_task.priority > current_task.priority:
                # Preempt current task and execute higher priority task
                remaining_prep_time = highest_priority_task.remaining_prep_time - (time_elapsed - highest_priority_task.arrival_time)
                current_task.remaining_prep_time -= remaining_prep_time
                time_elapsed += remaining_prep_time
                completed_tasks.append(current_task)
                current_task = highest_priority_task
                tasks.pop(0)
                continue

        # Simulate cooking time
        time_elapsed += current_task.remaining_cook_time

        # Check if there are higher priority tasks arrived while cooking
        if tasks and tasks[0].arrival_time <= time_elapsed:
            highest_priority_task = tasks[0]
            if highest_priority_task.priority > current_task.priority:
                # Preempt current task and execute higher priority task
                remaining_cook_time = highest_priority_task.remaining_cook_time - (time_elapsed - highest_priority_task.arrival_time)
                current_task.remaining_cook_time -= remaining_cook_time
                time_elapsed += remaining_cook_time
                completed_tasks.append(current_task)
                current_task = highest_priority_task
                tasks.pop(0)
                continue

        # Simulate attention time
        time_elapsed += current_task.remaining_attention_time

        # Increment priority after completing the task
        current_task.priority += 1
        completed_tasks.append(current_task)
        current_task = None

    return completed_tasks


if __name__ == "__main__":
    # Example kitchen tasks
    tasks = [
        KitchenTask(1, 0, 10, 5, 15),
        KitchenTask(2, 1, 8, 3, 20),
        KitchenTask(3, 2, 12, 4, 18)
    ]

    completed_tasks = kitchen_task_scheduling(tasks)

    print("Completed Kitchen Tasks:")
    for task in completed_tasks:
        print(task)
