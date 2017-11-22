%this script demos the usage of evaluation routines
% the result file 'demo.val.pred.txt' on validation data is evaluated
% against the ground truth

fprintf('MINI PLACES SCENE CLASSIFICATION CHALLENGE\n');

close all, clear all, clc;

%pred_file='demo.val.pred.txt';
%pred_file='alexnet-3100-val.txt';
%pred_file='alexnet-5000-val.txt';
%pred_file ='alexnet_bn-100-test.txt';
pred_file ='resnet34_base-32-test.txt';

ground_truth_file='../data/val.txt';
%ground_truth_file='../data/test.txt';
num_predictions_per_image=5;

fprintf('pred_file: %s\n', pred_file);
fprintf('ground_truth_file: %s\n', ground_truth_file);

error_cls = eval_cls(pred_file,ground_truth_file,1:num_predictions_per_image);

disp('# guesses vs cls error');
disp([(1:num_predictions_per_image)',error_cls']);
