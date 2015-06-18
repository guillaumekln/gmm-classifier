clear;
load 'data.mat';

train = double(reshape(TRAIN, 60000, 28*28)');
test = double(reshape(TEST, 10000, 28*28)');

nb_class = 10;
nb_class_comp = 2;

gmm = GMMClassifier(nb_class, nb_class_comp, LABEL_TRAIN, train);
gmm.showAllMeans();
taux = gmm.test(LABEL_TEST, test);

fprintf('Recognition rate: %f\n', taux);