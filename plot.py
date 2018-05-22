#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/21
import matplotlib.pyplot as plt
import logger

# ["epoch", "Lr", "Train Loss", "Valid Loss"]
data = logger.read_log('./log/log.csv')

plt.plot(data['epoch'], data['Lr'], label='lr')

# note the key "Train Loss", the space replaced by "_"
plt.semilogy(data['epoch'], data['Train_Loss'], label='training loss')
plt.semilogy(data['epoch'], data['Valid_Loss'], label='valid loss')

plt.grid()
plt.legend()
plt.show()
