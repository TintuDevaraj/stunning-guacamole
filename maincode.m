clc
clear all 
close all
[num,txt,raw]= xlsread('dataset.xlsx');
f = figure
t = uitable(f, 'Data', raw,'Position', [40 40 500 500]);
data=num;
data=num(:,1:62);
labelData=num(:,22);
X=data;
Y=labelData;