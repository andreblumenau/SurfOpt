#SurfOpt + AMSgrad +LogLoss

library(RcmdrMisc)
library(sp)
library(compiler)

#Clear memory
rm(list=ls())

#Call garbage collector
gc()
#############################################
b_to_x <- function(b,theta,c_param,d_param,x){
	therm = (sin(theta)*x^2*exp(sin(theta)*x^2*exp(b)+b))/(exp(sin(theta)*x^2*exp(b))+exp(cos(theta)*x+c_param))
	
	if(!is.finite(therm)||is.infinite(therm)){return(0)}
	return(therm)
}

b_to_y <- function(b,theta,c_param,d_param,x){
	therm = -(cos(theta)*x^2*exp(b))/(exp(cos(theta)*x^2*exp(b)+sin(theta)*x+d_param)+1)

	if(!is.finite(therm)||is.infinite(therm)){return(0)}
	return(therm)
}

t_to_x <- function(b,theta,c_param,d_param,x){
	therm = (x*exp(exp(b)*x^2*sin(theta))*(sin(theta)+exp(b)*x*cos(theta)))/(exp(exp(b)*x^2*sin(theta))+exp(x*cos(theta)+c_param))

	if(!is.finite(therm)||is.infinite(therm)){return(0)}
	return(therm)
}

t_to_y <- function(b,theta,c_param,d_param,x){
	therm = (x*(exp(b)*x*sin(theta)-cos(theta)))/(exp(x*sin(theta)+exp(b)*x^2*cos(theta)+d_param)+1)
	
	if(!is.finite(therm)||is.infinite(therm)){return(0)}
	return(therm)
}

c_to_x <- function(b,theta,c_param,d_param,x){
	therm = -exp(exp(b)*sin(theta)*x^2)/(exp(c_param+cos(theta)*x)+exp(exp(b)*sin(theta)*x^2))

	if(!is.finite(therm)||is.infinite(therm)){return(0)}
	return(therm)
}

d_to_y <- function(b,theta,c_param,d_param,x){
	therm = -1/(exp(d_param+exp(b)*cos(theta)*x^2+sin(theta)*x)+1)
	
	if(!is.finite(therm)||is.infinite(therm)){return(0)}
	return(therm)
}

################################################
Sigmoid <- function(x){
	return(1/(1+exp(-x)))
}

LogLoss <- function(x){
	return(-log(Sigmoid(x)))#Simplified Logistic Loss acts as if every example as 1
}

f_of_x <- function(x,smooth_ratio){
	return (x^2*exp(smooth_ratio))
}

rotate2 <- function(x,add_x,add_y,angle_theta,smooth_ratio){
	f_x <- f_of_x(x,smooth_ratio)
	return (rotate_point(x,f_x,add_x,add_y,angle_theta))
}

rotate_point <- function(x,y,add_x,add_y,angle_theta){
	x2 <- x*cos(angle_theta)-y*sin(angle_theta) + add_x
	y2 <- y*cos(angle_theta)+x*sin(angle_theta) + add_y
	
	if(!is.finite(x2) || !is.finite(y2)){
		print(paste("x =",x,"y =",y," add_x = ",add_x,"add_y =",add_y,"angle_theta = ",angle_theta))
	}
	
	return(c(x2,y2))
}

get_rotated_curve <- function(max_x,max_y,add_x,add_y,angle_theta,smooth_ratio,number_of_points){
	max_value = max(max_x,max_y)
	
	x = seq(from = -1*2*max_value, to = max_value*2, by = (max_value/number_of_points))
	y = seq(from = -1*2*max_value, to = max_value*2, by = (max_value/number_of_points))
	
	for( i in 1:(length(x))){			
			tuple <- rotate2(x[i],add_x,add_y,angle_theta,smooth_ratio)
			x[i] <- tuple[1]
			y[i] <- tuple[2]
	}		

	permited = which(y >= -3*max_value & y <= 3*max_value)	
		
	matriz <- matrix(0L,ncol=2,nrow=length(permited))
	matriz[,1] <- x[permited]
	matriz[,2] <- y[permited]	

	return (matriz)
}

f1_score <- function(perf,tp,tn,fp,fn){	

	if(min(tp,tn,fp,fn)<=0){
		return(0)
	}

	plus_class_precision = perf[1,1]/(perf[1,1]+perf[1,2])
	plus_class_recall = perf[1,1]/(perf[1,1]+perf[2,1])
	minus_class_precision = perf[2,2]/(perf[2,2]+perf[2,1])
	minus_class_recall = perf[2,2]/(perf[2,2]+perf[1,2])
	
	avg_precision = (plus_class_precision+minus_class_precision)/2	
	avg_recall = (plus_class_recall + minus_class_recall)/2
	
	f1_score_value = 2*((avg_precision*avg_recall)/(avg_precision+avg_recall))
	return(f1_score_value)
}

matthews_correlation_coefficient <- function(tp,tn,fp,fn){		
		#if(min(tp,tn,fp,fn)<=0){
		#	return(0)
		#}
		
		pp = tp+fp #predicted positive
		cp = tp+fn #condition positive
		pp = tn+fn #predicted negative
		cp = tn+fp #condition negative
		
		pp = tp+fp #predicted positive
		cp = tp+fn #condition positive
		pp = tn+fn #predicted negative
		cp = tn+fp #condition negative
		
		
		M <- (tp*tn - (fp*fn))/(sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.0000001)
		return(M)
}

random_coordinate <- function(coordinates,deviations){
	 return(c(mean(coordinates) + runif(1, -1*deviations,1*deviations)*sd(coordinates)))
	 #return(mean(coordinates))
}

random_angle <- function(){
	return(runif(1,-2*pi,2*pi))
}


#table_columns <- c("ADD_X","ADD_Y","ANGLE","SMTH","ACC","NPV","TNR","DistanceFromIdeal")
classify_with_curve <- function(x_coordinates,y_coordinates,class_of_points,add_x,add_y,angle_theta,smooth_ratio,number_of_points){
	
	evaluation = make_evaluation_matrix()
	
	max_actual_x = max(x_coordinates)
	max_actual_y = max(y_coordinates)
	curve_points = get_rotated_curve(max_actual_x,max_actual_y,add_x,add_y,angle_theta,smooth_ratio,number_of_points)	
	evaluation = curve_loss2(evaluation,x_coordinates,y_coordinates,class_of_points,add_x,add_y,angle_theta,smooth_ratio)
	return(evaluation)
}

classify_with_heuristic <- function(x_coordinates,y_coordinates,class_of_points,parameters,number_of_points){		
	max_actual_x = max(x_coordinates)
	max_actual_y = max(y_coordinates)	
	evaluation = make_evaluation_matrix()
	evaluation = curve_loss2(evaluation,x_coordinates,y_coordinates,class_of_points,parameters[,"ADD_X"],parameters[,"ADD_Y"],parameters[,"ANGLE"],parameters[,"SMTH"])
	
	evaluation 	 = cbind(evaluation,parameters[,"ITER"])
	colnames(evaluation) <- c( "ACC","NPV","TNR","TP","TN","FP","FN","MCC","F1","ERROR","ITER")
	return(evaluation)
}


transform_points <- function(x_coordinates,y_coordinates,add_x,add_y,angle_theta){	
	angle_theta = angle_theta*-1
	
	x_coordinates = x_coordinates*cos(angle_theta)-y_coordinates*sin(angle_theta) - add_x
	y_coordinates = y_coordinates*cos(angle_theta)+x_coordinates*sin(angle_theta) - add_y
	
	return(cbind(x_coordinates,y_coordinates))
}

make_evaluation_matrix <- function(){
	#initialize evaluation_matrix
	column_names <- c("ACC","NPV","TNR","TP","TN","FP","FN","MCC","F1","ERROR")
	evaluation = matrix(0L,ncol=(length(column_names)),nrow=1)
	colnames(evaluation) <- column_names
	
	return(evaluation)
}

curve_loss2 <- function(evaluation,x_coordinates,y_coordinates,class_of_points,add_x,add_y,angle_theta,smooth_ratio){

	transformed_points = transform_points(x_coordinates,y_coordinates,add_x,add_y,angle_theta)
	y_curve_equivalent = f_of_x(transformed_points[,1],smooth_ratio)

	predict_customized = matrix(0L, nrow =(length(x_coordinates)), ncol=1)
	#points(transformed_points[,1],transformed_points[,2], col="green",bg="green",pch=24)#"dodgerblue2"	
	indexes = which(transformed_points[,2] > y_curve_equivalent)
	
	predict_customized[indexes,1] = 1

	perf <- table(predict_customized,class_of_points)
	evaluation[,"ACC"] = 0
	evaluation[,"NPV"] = 0
	evaluation[,"TNR"] = 0
	evaluation[,"TP"] = 0
	evaluation[,"TN"] = 0
	evaluation[,"FP"] = 0
	evaluation[,"FN"] = 0
	evaluation[,"MCC"] = 0
	evaluation[,"F1"] = 0
	evaluation[,"ERROR"] = 0#sum(aaaaaa)/length(transformed_points)
	
	if (nrow(perf) > 1 && ncol(perf)>1){		
		evaluation[,"ACC"] = (perf[1,1]+perf[2,2])/length(predict_customized)

		evaluation[,"TP"] = as.double(perf[1,1]) #true positive
		evaluation[,"TN"] = as.double(perf[2,2]) #true negative
		evaluation[,"FP"] = as.double(perf[1,2]) #false positive
		evaluation[,"FN"] = as.double(perf[2,1]) #false negative
		
		evaluation[,"NPV"] = evaluation[,"TN"]/(evaluation[,"FN"]+evaluation[,"TN"]+0^(evaluation[,"FN"]+evaluation[,"TN"]))
		evaluation[,"TNR"] = evaluation[,"TN"]/(evaluation[,"FP"]+evaluation[,"TN"]+0^(evaluation[,"FP"]+evaluation[,"TN"]))
		
	}else{
		evaluation[,"ACC"] = perf[1,1]/length(predict_customized)
		evaluation[,"TP"] =  as.double(perf[1,1])
		if(ncol(perf)>1){
			perf = matrix(c(perf[1,1],0,perf[1,2],0),nrow=2)
			evaluation[,"TP"] =  as.double(perf[1,1])
			evaluation[,"FP"] =  as.double(perf[1,2])
		}else{
			perf = matrix(c(perf[1,1],0,0,0),nrow=2)
			evaluation[,"TP"] =  as.double(perf[1,1])
		}
	}

	evaluation[,"MCC"] = matthews_correlation_coefficient(evaluation[,"TP"],evaluation[,"TN"],evaluation[,"FP"],evaluation[,"FN"])
	evaluation[,"F1"] = 0#f1_score(perf,tp,tn,fp,fn)
	
	return(evaluation)
}

SurfOpt <- function(actual_points_x,actual_points_y,class_of_points,add_x,add_y,angle_theta,smooth_ratio,alpha_parameter,tnr_thresold,tpr_thresold,maxIterations,number_of_points){	
	
	max_actual_x 	= max(actual_points_x)
	max_actual_y 	= max(actual_points_y)
	curve_points 	= get_rotated_curve(max_actual_x,max_actual_y,add_x,add_y,angle_theta,smooth_ratio,number_of_points)

	instrumental_distance = 0 #"Weighted distance to ideal classifier
	infinitesimal = 0.0001 #add for numerical_estability aka prevents zero from blowing everything up
	last_iteration = maxIterations #may change on execution
	
	#patience_count = 0
	#patience_limit = 10
	
	coordinate_change = 1
	
	add_x = (add_x)
	add_y = (add_y)
	angle_theta =  (angle_theta)
	smooth_ratio = (smooth_ratio)
	smooth_ratio_copy = (smooth_ratio)

	bootstrap_const = (0.001)
	
	#parameters
	beta_parameter  = 0.9
	gamma_parameter = 0.999
	
	#Historical squared gradients for each variable
	historical_x 	      = bootstrap_const
	historical_y 	      = bootstrap_const
	historical_angle      = bootstrap_const
	historical_smoothness = bootstrap_const

	#Historical gradients for each variable
	historical_x_m 	   		= bootstrap_const
	historical_y_m 	   		= bootstrap_const
	historical_angle_m 		= bootstrap_const
	historical_smoothness_m = bootstrap_const
	
	#Parameter updates for each variable
	historical_update_x 		 = bootstrap_const
	historical_update_y 		 = bootstrap_const
	historical_update_angle 	 = bootstrap_const
	historical_update_smoothness = bootstrap_const
	
	#Update for each variable
	update_x = (0)
	update_y = (0)
	update_angle = (0)
	update_smoothness = (0)

	table_columns <- c("ADD_X","ADD_Y","ANGLE","SMTH","ACC","NPV","TNR","LAST","TP","TN","FP","FN","SIG","MCC","F1")
	result_table = matrix(0L, nrow = maxIterations,ncol=(length(table_columns)))	
	colnames(result_table) <- table_columns

	c_to_xCmp <- cmpfun(c_to_x)
	d_to_yCmp <- cmpfun(d_to_y)
	t_to_xCmp <- cmpfun(t_to_x)
	t_to_yCmp <- cmpfun(t_to_y)
	b_to_xCmp <- cmpfun(b_to_x)
	b_to_yCmp <- cmpfun(b_to_y)
	LogLossCmp <- cmpfun(LogLoss)
	
	#initialize error
	error_value = 0
	
	#initialize evaluation_matrix
	evaluation = make_evaluation_matrix()
	
	for( i in 1:maxIterations){
		curve_points = get_rotated_curve(max_actual_x,max_actual_y,add_x,add_y,angle_theta,smooth_ratio,number_of_points)
		evaluation = curve_loss2(evaluation,actual_points_x,actual_points_y,class_of_points,add_x,add_y,angle_theta,smooth_ratio)

		result_table[i,] = c(add_x,add_y,angle_theta,smooth_ratio,evaluation[,"ACC"],evaluation[,"NPV"],evaluation[,"TNR"],i,evaluation[,"TP"],evaluation[,"TN"],evaluation[,"FP"],evaluation[,"FN"],error_value,evaluation[,"MCC"],evaluation[,"F1"])				

		error_value = LogLossCmp(evaluation[,"ACC"])

		TP = evaluation[,"TP"]
		FN = evaluation[,"FN"]
		
		TPR = TP/((TP+FN)+0^(TP+FN))
		
		if(evaluation[,"TNR"] <= tnr_thresold || TPR <= tpr_thresold || nrow(curve_points) < 3){
		#if(patience_count == patience_limit || (nrow(curve_points)==0)){
			add_x = random_coordinate(actual_points_x,1)
			add_y = random_coordinate(actual_points_y,1)
			angle_theta = random_angle()
			smooth_ratio = smooth_ratio_copy	

			historical_x 	 	  = bootstrap_const
            historical_y 	 	  = bootstrap_const
            historical_angle 	  = bootstrap_const
			historical_smoothness = bootstrap_const
			
			historical_x_m 	   		= bootstrap_const
			historical_y_m 	   		= bootstrap_const
			historical_angle_m 		= bootstrap_const
		    historical_smoothness_m = bootstrap_const
			
			next
		}

		#reset gradient variables
		gradient_x 			= 0
		gradient_y 			= 0
		gradient_smoothness = 0
		gradient_angle 		= 0		
		
		#iteration over sample points from the curve
		iters = nrow(curve_points)
		for(j in 1:iters){
			x = curve_points[j,1]
			y = curve_points[j,2]
        
			term_grad_x = c_to_xCmp(smooth_ratio,angle_theta,add_x,add_y,x)
			term_grad_y = d_to_yCmp(smooth_ratio,angle_theta,add_x,add_y,x)
			term_grad_theta_1 = t_to_xCmp(smooth_ratio,angle_theta,add_x,add_y,x)
			term_grad_theta_2 = t_to_yCmp(smooth_ratio,angle_theta,add_x,add_y,x)
			term_grad_smothness_1 = b_to_xCmp(smooth_ratio,angle_theta,add_x,add_y,x)
			term_grad_smothness_2 = b_to_yCmp(smooth_ratio,angle_theta,add_x,add_y,x)		
		
			gradient_x = log(exp(gradient_x) * exp(term_grad_x))
			gradient_y = log(exp(gradient_y) * exp(term_grad_y))
			gradient_angle = log(exp(gradient_angle) * exp(term_grad_theta_1) * exp(term_grad_theta_2))
			gradient_smoothness = log(exp(gradient_smoothness)* exp(term_grad_smothness_1) * exp(term_grad_smothness_2))
		}
		
		#compute gradients
		gradient_x 			= log(exp(gradient_x)^error_value)
		gradient_y 			= log(exp(gradient_y)^error_value)
		gradient_smoothness = log(exp(gradient_smoothness)^(error_value))
		gradient_angle 		= log(exp(gradient_angle)^(error_value))
		
		#accumulate with infinity norm squared gradients
		g_complement = 1-gamma_parameter
		
				
		#accumulate exponential decaying average gradients
		b_complement = 1-beta_parameter
		historical_x_m 	   		= beta_parameter*historical_x_m + 		   gradient_x*b_complement
		historical_y_m 	   		= beta_parameter*historical_y_m + 		   gradient_y*b_complement
		historical_angle_m 		= beta_parameter*historical_angle_m + 	   gradient_angle*b_complement
		historical_smoothness_m = beta_parameter*historical_smoothness_m + gradient_smoothness*b_complement			
		
		historical_x 	 	  = max(historical_x,gamma_parameter*historical_x + log(exp(gradient_x^2)^g_complement))
		historical_y 	 	  = max(historical_y,gamma_parameter*historical_y + log(exp(gradient_y^2)^g_complement))
		historical_angle 	  = max(historical_angle,gamma_parameter*historical_angle + log(exp(gradient_angle^2)^g_complement))
		historical_smoothness = max(historical_smoothness,gamma_parameter*historical_smoothness + log(exp(gradient_smoothness^2)^g_complement))
		
		#compute updates		
		update_x 		  = (alpha_parameter/(sqrt(historical_x)+infinitesimal))*historical_x_m
		update_y 		  = (alpha_parameter/(sqrt(historical_y)+infinitesimal))*historical_y_m
		update_angle 	  = (alpha_parameter/(sqrt(historical_angle)+infinitesimal))*historical_angle_m
		update_smoothness = (alpha_parameter/(sqrt(historical_smoothness)+infinitesimal))*historical_smoothness_m
		
		#avoids NaN and crashes
		if(!is.finite(update_x)){update_x 					= infinitesimal}
		if(!is.finite(update_y)){update_y 					= infinitesimal}
		if(!is.finite(update_angle)){update_angle 			= infinitesimal}
		if(!is.finite(update_smoothness)){update_smoothness = infinitesimal}		
		
		#apply updates
		angle_theta  = angle_theta 	- log(exp(update_angle)^(angle_theta))
		smooth_ratio = smooth_ratio - log(exp(update_smoothness)^(smooth_ratio))
		add_x 		 = add_x 		- log(exp(update_x)^(add_x))
		add_y 		 = add_y 		- log(exp(update_y)^(add_y))
	}
	return(result_table[1:last_iteration,])
}

center_in_origin <- function(coordinates_vector,indexes){
	median_sold = median(coordinates_vector[indexes,])
	coordinates_vector = coordinates_vector - median_sold
	return(coordinates_vector)
}

#Filter out curves that shown ACC,NPV,TNR in decreasing order.
heuristic_on_metrics <- function(table_data){
	r_value = which(table_data[,"ACC"]> table_data[,"NPV"] && table_data[,"NPV"]> table_data[,"TNR"])
	if(length(r_value)==0){
		return(1:nrow(table_data))
	}
	return(r_value)
}

heuristic_max <- function(table_data, colum_name){
	indexes_heuristic = heuristic_on_metrics(table_data)
	best_value = max(table_data[indexes_heuristic,colum_name])
	best_index = which(table_data[,colum_name]==best_value)
	best_index = best_index[1]
	columns = c("ADD_X","ADD_Y","ANGLE","SMTH")	
	row_result = c(as.numeric(table_data[best_index,columns]),best_index)
	row_result = t(as.matrix(row_result))
	colnames(row_result) <- c(columns,"ITER")	
	return(row_result)
}

heuristic_min <- function(table_data, colum_name){
	indexes_heuristic = heuristic_on_metrics(table_data)
	best_value = min(table_data[indexes_heuristic,colum_name])
	best_index = which(table_data[,colum_name]==best_value)
	best_index = best_index[1]
	columns = c("ADD_X","ADD_Y","ANGLE","SMTH")	
	row_result = c(as.numeric(table_data[best_index,columns]),best_index)
	row_result = t(as.matrix(row_result))
	colnames(row_result) <- c(columns,"ITER")
	return(row_result)
}

heuristic_max_sum <- function(table_data, cols_to_sum){
	indexes_heuristic = heuristic_on_metrics(table_data)
	
	best_value = 0
	if(nrow(table_data[indexes_heuristic,cols_to_sum])>1){	
		best_value = max(rowSums(table_data[indexes_heuristic,cols_to_sum]))
		best_index = which(rowSums(table_data[,cols_to_sum])==best_value)
		best_index = best_index[1]
	}
	
	if(nrow(table_data[indexes_heuristic,cols_to_sum])==1){	
		best_value = rowSums(table_data[indexes_heuristic,cols_to_sum])
		best_index = which(rowSums(table_data[,cols_to_sum])==best_value)
		best_index = best_index[1]
	}
	
	columns = c("ADD_X","ADD_Y","ANGLE","SMTH")	
	row_result = c(as.numeric(table_data[best_index,columns]),best_index)
	row_result = t(as.matrix(row_result))
	colnames(row_result) <- c(columns,"ITER")	
	return(row_result)
}

heuristic_max_sum_scaled <- function(table_data, cols_to_sum){
	indexes_heuristic = heuristic_on_metrics(table_data)
	best_value = max(rowSums(scale(table_data[indexes_heuristic,cols_to_sum])))
	best_index = which(rowSums(scale(table_data[,cols_to_sum]))==best_value)
	best_index = best_index[1]
	columns = c("ADD_X","ADD_Y","ANGLE","SMTH")	
	row_result = c(as.numeric(table_data[best_index,columns]),best_index)
	row_result = t(as.matrix(row_result))
	colnames(row_result) <- c(columns,"ITER")
	return(row_result)
}

plot_points_2D_simpler <- function(x_coordinates,y_coordinates,classB,classA,min_x,max_x,min_y,max_y,main_title="Majority x Minority"){	
	some_red = "#FF3300"
	some_blue = "#0081FF" #"#0000CB" #
	
	plot(x_coordinates[classB],y_coordinates[classB], xlab="", ylab="", main=main_title, ylim=c(min_y,max_y), xlim=c(min_x,max_x), pch=25, col="#FF3300",bg="#FF3300",xaxt='n',yaxt='n')

	points(x_coordinates[classA],y_coordinates[classA], col="dodgerblue2",bg="dodgerblue2",pch=24)#"dodgerblue2"	
}