#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INITIAL_WEIGHT 0.3
#define NETWORK_WEIGHT_NUM 9
#define LEARNING_RATE 0.6
#define ERROR_MAX 0.008
#define BIAS_UNION 1.0


typedef struct tag_TrainSet
{
	double x_train_data[2];
	double y_train_label;
}TrainSample;

double g_weight[NETWORK_WEIGHT_NUM];
double g_hidden_1_out, g_hidden_2_out;
double g_out;

void network_initial(void);
double sigmaFun(double z);
double forwardFun(TrainSample *trainset);
double error_final = 10;
int BackPropagation(TrainSample *trainset, int trainset_num);


int main()
{
	int i;
	TrainSample train[4];

	unsigned int iteration = 0;

	train[0].x_train_data[0] = 0.0;
	train[0].x_train_data[1] = 0.0;
	train[0].y_train_label = 0.0;

	train[1].x_train_data[0] = 0.0;
	train[1].x_train_data[1] = 1.0;
	train[1].y_train_label = 1.0;

	train[2].x_train_data[0] = 1.0;
	train[2].x_train_data[1] = 0.0;
	train[2].y_train_label = 1.0;

	train[3].x_train_data[0] = 1.0;
	train[3].x_train_data[1] = 1.0;
	train[3].y_train_label = 0.0;

	network_initial();

	while (error_final > 0.008)
	{
		printf("iteration %u\n", iteration);
		iteration++;
		BackPropagation(train, 4);

		/*for (i = 0; i < NETWORK_WEIGHT_NUM; i++)
		{
			printf("%lf\n", g_weight[i]);
		}*/
	
	}

	for (i = 0; i < NETWORK_WEIGHT_NUM; i++)
	{
		printf("%lf\n", g_weight[i]);
	}
	printf("iteration %u\n", iteration);
	system("pause");
	return 0;
}

void network_initial(void)
{
	g_weight[0] = 0.0543;
	g_weight[1] = 0.0579;
	g_weight[2] = -0.0291;
	g_weight[3] = 0.0999;
	g_weight[4] = 0.0801;
	g_weight[5] = -0.0605;
	g_weight[6] = -0.0703;
	g_weight[7] = -0.0939;
	g_weight[8] = -0.0109;
	g_hidden_1_out = 0;
	g_hidden_2_out = 0;
	g_out = 0;
}

double sigmaFun(double z)
{
	return 1.0 / (1.0 + exp(-z));
}

double forwardFun(TrainSample *trainset)
{
	double hidden_1_input;
	double hidden_2_input;
	double out_input;
	double error_return;

	hidden_1_input = BIAS_UNION * g_weight[6] + trainset->x_train_data[0] * g_weight[0] + trainset->x_train_data[1] * g_weight[2];
	g_hidden_1_out = sigmaFun(hidden_1_input);
	hidden_2_input = BIAS_UNION * g_weight[7] + trainset->x_train_data[0] * g_weight[1] + trainset->x_train_data[1] * g_weight[3];
	g_hidden_2_out = sigmaFun(hidden_2_input);
	out_input = BIAS_UNION * g_weight[8] + g_hidden_1_out * g_weight[4] + g_hidden_2_out * g_weight[5];
	g_out = sigmaFun(out_input);


	error_return = 0.5 * (g_out - trainset->y_train_label) * (g_out - trainset->y_train_label);
	return error_return;
}

int BackPropagation(TrainSample *trainset, int trainset_num)
{
	double error_out = 0.0;
	double diff_error_weight[NETWORK_WEIGHT_NUM];
	int i, j;
	
	
	for (i = 0; i < NETWORK_WEIGHT_NUM; i++)
	{
		diff_error_weight[i] = 0;
	}

	 
	for (i = 0; i < trainset_num; i++)
	{
		error_out += forwardFun((trainset + i));
		diff_error_weight[8] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out);
		diff_error_weight[4] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_hidden_1_out;
		diff_error_weight[5] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_hidden_2_out;
		diff_error_weight[6] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_weight[4] * g_hidden_1_out * (1 - g_hidden_1_out);
		diff_error_weight[7] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_weight[5] * g_hidden_2_out * (1 - g_hidden_2_out);
		diff_error_weight[0] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_weight[4] * g_hidden_1_out * (1 - g_hidden_1_out) * trainset[i].x_train_data[0];
		diff_error_weight[1] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_weight[5] * g_hidden_2_out * (1 - g_hidden_2_out) * trainset[i].x_train_data[0];
		diff_error_weight[2] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_weight[4] * g_hidden_1_out * (1 - g_hidden_1_out) * trainset[i].x_train_data[1];
		diff_error_weight[3] = (g_out - trainset[i].y_train_label) * g_out * (1 - g_out) * g_weight[5] * g_hidden_2_out * (1 - g_hidden_2_out) * trainset[i].x_train_data[1];

		for (j = 0; j < NETWORK_WEIGHT_NUM; j++)
		{
			g_weight[j] -= LEARNING_RATE * diff_error_weight[j];
		}
	}
	error_final = error_out / trainset_num;
	printf("error_final = %lf\n", error_final);

	return 0;
}
