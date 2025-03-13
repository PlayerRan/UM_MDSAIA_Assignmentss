def solve(teams, values):
    # Split the teams into two groups
    group1 = teams[:4]
    group2 = teams[4:]
    values1 = values[:4]
    values2 = values[4:]

    # Find the winners of each group
    winner1 = group1[values1.index(max(values1))]
    winner2 = group2[values2.index(max(values2))]
    sorted_values1 = sorted(values1, reverse=True)
    sorted_values2 = sorted(values2, reverse=True)
    runner_up1 = group1[values1.index(sorted_values1[1])]
    runner_up2 = group2[values2.index(sorted_values2[1])]

    # Determine the champion and runner-up
    if max(values1) > max(values2):
        champion = winner1
        runner_up = max(runner_up1, winner2, key=lambda team: values[teams.index(team)])
    else:
        champion = winner2
        runner_up = max(runner_up2, winner1, key=lambda team: values[teams.index(team)])

    print(f"{champion} {runner_up}")

teams = []
values = []
for i in range(8):
    input_data = input().strip().split()
    teams.append(input_data[0])
    values.append(int(input_data[1]))
solve(teams, values)