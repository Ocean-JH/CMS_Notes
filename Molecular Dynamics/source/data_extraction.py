#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Read the file and search for lines containing the pattern "Cohesive Energy of Ar a = * E = *
import re

# Initialize list to hold the matching lines
matching_lines = []

# Define the regex pattern for capturing a and E values
pattern = re.compile(r"Cohesive Energy of Cu v = ([\d\.-]+)\s+x = ([\d\.-]+)\s+E = ([\d\.-]+)")

# Read the file and search for the pattern
with open('./log.lammps', 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            v_value = float(match.group(1))
            E_value = float(match.group(3))
            matching_lines.append((v_value, E_value))

# Write the extracted a and E values to a CSV file
with open('Cu_lj.csv', 'w') as f:
    f.write("v,E\n")  # Write header
    for v, E in matching_lines:
        f.write(f"{v},{E}\n")
