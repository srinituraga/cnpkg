load ~/data/E1088/validate4.mat im comp*
compTrue=compSkel;
%compTrue=comp;

conn261=hdf5read('/home/sturaga/exhibit/261/conn_e1088_train_validate4','/main');
conn607=hdf5read('/home/sturaga/exhibit/607/conn_e1088_train_validate4','/main');
conn608=hdf5read('/home/sturaga/exhibit/608/conn_e1088_train_validate4','/main');
clearvars -except comp compTrue conn* im set
nets = { ...
	'261', ...
	'607', ...
	'608'};
legendStr = { ...
	'Minimax 261 (6 layer)', ...
	'Minimax 607 (6 layer)', ...
	'Minimax 608 (6 layer)'};

th=[0.1:0.1:0.4 0.42:0.02:0.5 0.51:.01:0.74 0.75:0.05:.95 .97 .99 .999];

compute_metrics
save metrics_validate4

clear all

load ~/data/E1088/roi7.mat im comp*
compTrue=compSkel;

conn261=hdf5read('/home/sturaga/exhibit/261/conn_e1088_test_roi7','/main');
conn607=hdf5read('/home/sturaga/exhibit/607/conn_e1088_test_roi7','/main');
conn608=hdf5read('/home/sturaga/exhibit/608/conn_e1088_test_roi7','/main');
clearvars -except comp compTrue conn* im set
nets = { ...
	'261', ...
	'607', ...
	'608'};
legendStr = { ...
	'Minimax 261 (6 layer)', ...
	'Minimax 607 (6 layer)', ...
	'Minimax 608 (6 layer)'};

th=[0.1:0.1:0.4 0.42:0.02:0.5 0.51:.01:0.74 0.75:0.05:.95 .97 .99 .999];

compute_metrics
save metrics_roi7

clear all

set = 'e2006_roi2'
load ~/data/E2006/semiautoROI2/roi2.mat im comp*
compTrue=comp;

conn261=hdf5read(['/home/sturaga/exhibit/261/conn_' set],'/main');
conn607=hdf5read(['/home/sturaga/exhibit/607/conn_' set],'/main');
conn608=hdf5read(['/home/sturaga/exhibit/608/conn_' set],'/main');
clearvars -except comp compTrue conn* im set
nets = { ...
	'261', ...
	'607', ...
	'608'};
legendStr = { ...
	'Minimax 261 (6 layer)', ...
	'Minimax 607 (6 layer)', ...
	'Minimax 608 (6 layer)'};

th=[0.1:0.1:0.4 0.42:0.02:0.5 0.51:.01:0.74 0.75:0.05:.95 .97 .99 .999];

compute_metrics
save(['metrics_' set])

clear all

set = 'e2006_roi3'
load ~/data/E2006/semiautoROI3/roi3.mat im comp*
compTrue=comp;

conn261=hdf5read(['/home/sturaga/exhibit/261/conn_' set],'/main');
conn607=hdf5read(['/home/sturaga/exhibit/607/conn_' set],'/main');
conn608=hdf5read(['/home/sturaga/exhibit/608/conn_' set],'/main');
clearvars -except comp compTrue conn* im set
nets = { ...
	'261', ...
	'607', ...
	'608'};
legendStr = { ...
	'Minimax 261 (6 layer)', ...
	'Minimax 607 (6 layer)', ...
	'Minimax 608 (6 layer)'};

th=[0.1:0.1:0.4 0.42:0.02:0.5 0.51:.01:0.74 0.75:0.05:.95 .97 .99 .999];

compute_metrics
save(['metrics_' set])

clear all

set = 'e2006_roi5'
load ~/data/E2006/semiautoROI5/roi5.mat im comp*
compTrue=comp;

conn261=hdf5read(['/home/sturaga/exhibit/261/conn_' set],'/main');
conn607=hdf5read(['/home/sturaga/exhibit/607/conn_' set],'/main');
conn608=hdf5read(['/home/sturaga/exhibit/608/conn_' set],'/main');
clearvars -except comp compTrue conn* im set
nets = { ...
	'261', ...
	'607', ...
	'608'};
legendStr = { ...
	'Minimax 261 (6 layer)', ...
	'Minimax 607 (6 layer)', ...
	'Minimax 608 (6 layer)'};

th=[0.1:0.1:0.4 0.42:0.02:0.5 0.51:.01:0.74 0.75:0.05:.95 .97 .99 .999];

compute_metrics
save(['metrics_' set])
