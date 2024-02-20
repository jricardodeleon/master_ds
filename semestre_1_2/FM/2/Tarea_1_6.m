{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red15\green112\blue16;\red8\green0\blue255;
\red148\green0\blue242;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c0\c50196\c7451;\cssrgb\c5490\c0\c100000;
\cssrgb\c65490\c3529\c96078;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs20 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 clc \cf3 \strokec3 % Limpia pantalla\cf2 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 % Generando un conjunto de datos D\cf2 \strokec2 \
x = 0:0.01:5; \cf3 \strokec3 % Crea un vector x de 0 a 5 con paso de incremento de 0.01\cf2 \strokec2 \
[m,n] = size(x); \cf3 \strokec3 % Obtener el tama\'f1o del vector m renglones y n columnas\cf2 \strokec2 \
y = 0.21*x + 0.05*rand(1,n);   \cf3 \strokec3 % Formando un vector y con base una recta pero con valores aleatorios \cf2 \strokec2 \
\cf3 \strokec3 % Separo datos \cf2 \strokec2 \
t = n - 100;\
x_training = x(1:t);  \cf3 \strokec3 % Estos son x para entrenar\cf2 \strokec2 \
y_training = y(1:t);  \cf3 \strokec3 % Estos son y para entrenar\cf2 \strokec2 \
[m,n] = size(x_training); \cf3 \strokec3 % Obtener el tama\'f1o del vector m renglones y n columnas de entrenamiento\cf2 \strokec2 \
eta = 0.06;\
w11 = rand(1,1);  \cf3 \strokec3 % Inicializa el peso en valor aleatorio  w(t0)\cf2 \strokec2 \
w11v = [];  \cf3 \strokec3 % Vector para ir guardando la evoluci\'f3n de los datos\cf2 \strokec2 \
y_redv = [];  \cf3 \strokec3 % Vector para ir guardando la evoluci\'f3n de los datos\cf2 \strokec2 \
\cf3 \strokec3 % Vamos a formar el ciclo de entrenamiento \cf2 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 for \cf2 \strokec2 i = 1:n\
    \cf3 \strokec3 % El perceptron\cf2 \strokec2 \
    z11 = w11*x_training(1,i);  \cf3 \strokec3 % Funci\'f3n de preactivaci\'f3n \cf2 \strokec2 \
    y_red = z11;    \cf3 \strokec3 % Funci\'f3n de activaci\'f3n (en este caso identidad)\cf2 \strokec2 \
    error = y_training(1,i) - y_red;  \cf3 \strokec3 % Error cometido\cf2 \strokec2 \
    \cf3 \strokec3 % Ley de actualizaci\'f3n\cf2 \strokec2 \
    delta_w = eta*error*x_training(1,i); \cf3 \strokec3 % Calcula el t\'e9rmino de ajuste\cf2 \strokec2 \
    w11v = [w11v w11];  \cf3 \strokec3 % Guardar el valor anterior en el vector\cf2 \strokec2 \
    y_redv = [y_redv y_red];\
    w11 = w11 + delta_w;  \cf3 \strokec3 % Actualiza el nuevo peso w(t+1)\cf2 \strokec2 \
\cf4 \strokec4 end\cf2 \strokec2 \
\
plot(x_training,y_training,x_training,y_redv)\
title(\cf5 \strokec5 "Trainng X Y"\cf2 \strokec2 )\
figure;\
plot(w11v)\
title(\cf5 \strokec5 "weight"\cf2 \strokec2 )\
\pard\pardeftab720\partightenfactor0
\cf3 \strokec3 %% Test\cf2 \strokec2 \
x_test = x(t+1:end);\
y_test = y(t+1:end);\
\cf3 \strokec3 % Usando la red\cf2 \strokec2 \
y_redtest = w11*x_test;\
figure;\
plot(x_test, y_test, x_test, y_redtest,\cf5 \strokec5 'b-'\cf2 \strokec2 )\
error_test = y_test - y_redtest ;\
title(\cf5 \strokec5 "Error Test"\cf2 \strokec2 )\
figure();\
plot(error)\
title(\cf5 \strokec5 "Error"\cf2 \strokec2 )\
\
\cf3 \strokec3 % xi yi nuevos puntos\cf2 \strokec2 \
xi = 0:0.01:5; \cf3 \strokec3 % Crea un vector x de 0 a 5 con paso de incremento de 0.01\cf2 \strokec2 \
[m,n] = size(x); \cf3 \strokec3 % Obtener el tama\'f1o del vector m renglones y n columnas\cf2 \strokec2 \
yi = 0.21 *x + 0.05*rand(1,n);   \cf3 \strokec3 %\cf2 \strokec2 \
figure;\
b = w11 * xi; \
scatter(xi,yi, marker=\cf5 \strokec5 'o'\cf2 \strokec2 ); hold \cf5 \strokec5 on\cf2 \strokec2 ;\
plot(xi,b, \cf5 \strokec5 "LineWidth"\cf2 \strokec2 ,4); hold \cf5 \strokec5 off\cf2 \strokec2 ;\
title(\cf5 \strokec5 "Tarea"\cf2 \strokec2 )\
\
}