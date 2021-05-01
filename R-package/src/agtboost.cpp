
/*
 * agtboost: Adaptive and automatic gradient tree boosting computations.
 * Berent Lunde
 * 07.09.2019
 */

#include "agtboost.hpp"



// ---------------- ENSEMBLE ----------------
ENSEMBLE::ENSEMBLE(){
    this->first_tree = NULL;
    this->nrounds = 5000;
    this->best_it = NULL;
    this->sample_rate = NULL;
    this->learning_rate=0.01;
    this->extra_param = 0.0;
    this->loss_function = "mse";
}

ENSEMBLE::ENSEMBLE(double learning_rate_){
    this->first_tree = NULL;
    this->nrounds = 5000;
    this->best_it = NULL;
    this->sample_rate = NULL;
    this->learning_rate=learning_rate_;
    this->extra_param = 0.0;
    this->loss_function = "mse";
}

void ENSEMBLE::set_param(int nrounds_, double learning_rate_, double extra_param_, std::string loss_function_)
{
    this->nrounds = nrounds_;
    this->learning_rate = learning_rate_;
    this->extra_param = extra_param_;
    this->loss_function = loss_function_;
}

int ENSEMBLE::get_nrounds(){
    return this->nrounds;
}

double ENSEMBLE::get_learning_rate(){
    return this->learning_rate;
}

double ENSEMBLE::get_extra_param(){
    return this->extra_param;
}

std::string ENSEMBLE::get_loss_function(){
    return this->loss_function;
}

void ENSEMBLE::serialize(ENSEMBLE *eptr, std::ofstream& f)
{
    // If current ENSEMBLE is NULL, return
    if(eptr == NULL)
    {
        //Rcpp::Rcout << "Trying to save NULL pointer" << std::endl;
        return;
    }
    
    f << std::fixed << eptr->nrounds << "\n";
    f << std::fixed << eptr->learning_rate << "\n";
    f << std::fixed << eptr->extra_param << "\n";
    f << std::fixed << eptr->initialPred << "\n";
    f << std::fixed << eptr->initial_score << "\n";
    f << eptr->loss_function << "\n";
    
    eptr->first_tree->serialize(eptr->first_tree, f);
    f.close();
    
}

void ENSEMBLE::deSerialize(ENSEMBLE *eptr, std::ifstream& f)
{
    
    // Check stream
    std::streampos oldpos = f.tellg();
    int val;
    int MARKER = -1;
    if( !(f >> val) || val==MARKER ){
        return;   
    }
    f.seekg(oldpos);
    
    // Read from stream
    f >> eptr->nrounds >> eptr->learning_rate >> eptr->extra_param >>
        eptr->initialPred >> eptr->initial_score >> eptr->loss_function;
    
    // Start recurrence
    int lineNum = 6;
    eptr->first_tree = new GBTREE;
    eptr->first_tree->deSerialize(eptr->first_tree, f, lineNum);
    
}

void ENSEMBLE::save_model(std::string filepath)
{
    std::ofstream f;
    f.open(filepath.c_str());
    this->serialize(this, f);
    f.close();
}
void ENSEMBLE::load_model(std::string filepath)
{
    std::ifstream f;
    f.open(filepath.c_str());
    this->deSerialize(this, f);
    f.close();
}

 
double ENSEMBLE::initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w){
    
    double pred=0;
    double pred_g_transform = y.sum()/w.sum(); // should be optim given weights...
    
    if(loss_function=="mse"){
        pred = pred_g_transform;
    }else if(loss_function=="logloss"){
        //double pred_g_transform = (y*w).sum()/n; // naive probability
        pred = log(pred_g_transform) - log(1 - pred_g_transform);
    }else if(loss_function=="poisson"){
        //double pred_g_transform = (y*w).sum()/n; // naive intensity
        pred = log(pred_g_transform);
    }else if(loss_function=="gamma::neginv"){
        //double pred_g_transform = (y*w).sum()/n;
        pred = - 1.0 / pred_g_transform;
    }else if(loss_function=="gamma::log"){
        pred = log(pred_g_transform);
    }else if(loss_function=="negbinom"){
        pred = log(pred_g_transform);
    }
    
    return pred;
}


void ENSEMBLE::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, 
                     bool force_continued_learning, Tvec<double> &w){
    // Set init -- mean
     int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> pred(n), g(n), h(n);
    
    // MSE -- FIX FOR OTHER LOSS FUNCTIONS
    this->initialPred = this->initial_prediction(y, loss_function, w); //y.sum()/n;
    pred.setConstant(this->initialPred);
    this->initial_score = loss(y, pred, loss_function, w, this); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    // PARAMETERS FOR CIR CONTROL: Choose nsim and nobs by user
    // Default to nsim=100 nobs=100
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = dloss(y, pred, loss_function, this) * w;
    h = ddloss(y, pred, loss_function, this) * w;
    //Rcpp::Rcout << g.array()/h.array() << std::endl;
    

    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism();
        //learning_rate_set * current_tree->getFeatureMapOptimism();
    if(verbose>0){
        Rcpp::Rcout  <<
            std::setprecision(4) <<
            "it: " << 1 << 
            "  |  n-leaves: " << current_tree->getNumLeaves() <<
            "  |  tr loss: " << loss(y, pred, loss_function, w, this) <<
            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
             std::endl;
    }
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, loss_function, this) * w;
        h = ddloss(y, pred, loss_function, this) * w;
        
        // Check perfect fit
        if(((g.array())/h.array()).matrix().maxCoeff() < 1e-12){
            // Every perfect step is below tresh
            break;
        }
        
        
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeOptimism();
            //1.0*learning_rate_set * new_tree->getFeatureMapOptimism();

        // Update preds -- if should not be updated for last iter, it does not matter much computationally
        pred = pred + learning_rate * (new_tree->predict_data(X));
            
        // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
        if(verbose>0){
            if(i % verbose == 0){
                Rcpp::Rcout  <<
                    std::setprecision(4) <<
                        "it: " << i << 
                        "  |  n-leaves: " << new_tree->getNumLeaves() << 
                        "  |  tr loss: " << loss(y, pred, loss_function, w, this) <<
                        "  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
                        std::endl;
                
            }
        }
        
        // Stopping criteria 
        // Check for continued learning
        if(!force_continued_learning){
            
            // No forced learning
            // Check criterion
            if(expected_loss > EPS){
                break;
            }
            
        }
        
        // Passed criterion or force passed: Update ensemble
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
        
    }
}




void ENSEMBLE::sample_train(Tvec<double> &y, Tmat<double> &X, int verbose, std::string gen_loss_type, 
                     Tvec<double> sample_rate, std::string step_type, bool greedy_complexities, 
                     bool force_continued_learning, Tvec<double> &w, bool exclude_bad,
                     bool include_constant, bool keep_tail, int max_max_no_red, std::string change_criteria, double seed){
    
    if(seed != 999){
        std::srand(seed);
    }
    int MAXITER = nrounds;
    int n = y.size();
    double learning_rate_set = this->learning_rate; 
    double sample_rate_orig = 0;

    double sample_rate_set = sample_rate[0];
    int MAX_NO_REDUCTION = max_max_no_red - max_max_no_red * (sample_rate_set*sample_rate_set*sample_rate_set);

    if(sample_rate_set == 999){
      sample_rate_orig = 999;
      float LO = 0.1;
      float HI = 1.0;
      sample_rate_set = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
      MAX_NO_REDUCTION = 50;
    }

    int counter = 0;
  
    bool increasing = false;
    double EPS = 1E-9;
    double prog_EPS = 1E-9;
    double total_opt = 0.0;
    double gen_loss = 0.0;
    double expected_loss;
    double gen_loss_change = 0.0;
    double last_change_ratio = 0.0;
    double best_gen_loss = 0.0;
    double new_initial_score = 0.0;
    double best_opt = 0.0;
    double R_opt = 0.0;
    double best_R_opt = 0.0;
    double opt_tmp = 0.0;
    double change = 0.0;
    
    double const_pred;
    Tvec<double> pred(n), best_pred(n), keep_pred(n), g(n), h(n);
    this->best_it = 1;
    
    
    Tvec<int> ind_sub = sample_int_rate(n, sample_rate_set);
    Tmat<double> X_sub = matrix_subset(X, ind_sub);
    Tvec<double> y_sub = sample_vec(y, ind_sub);

    this->initialPred = this->initial_prediction(y_sub, loss_function, w); //y.sum()/n;
    pred.setConstant(this->initialPred);

    Tvec<double> pred_sub = sample_vec(pred, ind_sub);

    this->initial_score = loss(y_sub, pred_sub, loss_function, w, this); //(y - pred).squaredNorm() / n;

    // Prepare cir matrix
    // PARAMETERS FOR CIR CONTROL: Choose nsim and nobs by user
    // Default to nsim=100 nobs=100
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = dloss(y, pred, loss_function, this) * w;
    h = ddloss(y, pred, loss_function, this) * w;
    //Rcpp::Rcout << g.array()/h.array() << std::endl;
    
    Tvec<double> g_sub = sample_vec(g, ind_sub);
    Tvec<double> h_sub = sample_vec(h, ind_sub);
    

    this->first_tree = new GBTREE;
    this->first_tree->train(g_sub, h_sub, X_sub, cir_sim, greedy_complexities, learning_rate_set);
    
    GBTREE* current_tree = this->first_tree;
    GBTREE* current_best_tree = this->first_tree;
    current_tree->sampleRate=sample_rate_set;
    Tvec<double> pred_tmp = current_tree->predict_data(X);

    pred = pred + learning_rate_set * pred_tmp;
    pred_sub = sample_vec(pred, ind_sub);

    total_opt += current_tree->getTreeOptimism()*learning_rate_set*sample_rate_set;
    best_opt = total_opt;
    R_opt = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism() + initial_score - loss(y_sub, pred_sub, loss_function, w, this);
    R_opt = sample_rate_set * R_opt;

    expected_loss = loss_gtb(g, h, pred) * learning_rate_set*(2-learning_rate_set) +
        sample_rate_set * learning_rate_set * current_tree->getTreeOptimism();

    
    if(verbose>0){
        Rcpp::Rcout  <<
            std::setprecision(4) <<
            "it: " << 1 << 
            //"  |  n-leaves: " << new_tree->getNumLeaves() << 
            "  |  tr loss: " << loss(y_sub, pred_sub, loss_function, w, this) <<
            "  |  R opt: " << R_opt <<
            //"  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
            // "  |  loss_gtb: " << expected_loss << 
            // "  |  optimism: " << opt_tmp << 
            // "  |  gen_loss: " << gen_loss << 
            // "  |  sampSize: " << sample_rate_set << 
            // "  |  best it: " << best_it <<  
             std::endl;
    }
    
    if(gen_loss_type == "R"){
            best_gen_loss = loss(y, pred, loss_function, w, this) + R_opt;
            gen_loss_change = gen_loss - best_gen_loss;
    }
    else if(gen_loss_type == "expected"){
                gen_loss_change += expected_loss;
    }
    else if(gen_loss_type == "actual"){
            best_gen_loss = loss(y, pred, loss_function, w, this) + total_opt;
            gen_loss_change = gen_loss - best_gen_loss;
    }

    int i = 2;
    int round = 0;
    bool improved = false;
    for(int j=0; j<(sample_rate.size()); j++){
        
        
        sample_rate_set = sample_rate[j];
        int MAX_NO_REDUCTION = max_max_no_red - max_max_no_red * (sample_rate_set*sample_rate_set*sample_rate_set);
        if(sample_rate_set == 999){
            MAX_NO_REDUCTION = max_max_no_red;
        }
        while(i<(MAXITER+1)){
            if(sample_rate_orig == 999){
                float LO = 0.1;
                float HI = 1.0;
                sample_rate_set = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
            }
            
            // check for interrupt every iterations
            if (i % 1 == 0)
                Rcpp::checkUserInterrupt();


            // TRAINING
            GBTREE* new_tree = new GBTREE();
            new_tree -> sampleRate = sample_rate_set;

            best_pred = pred;

            ind_sub = sample_int_rate(n, sample_rate_set);

            // Adding constant prediction between trees
            if(include_constant){
                g = dloss(y, pred, loss_function, this)*w;
                h = ddloss(y, pred, loss_function, this)*w;
                g_sub = sample_vec(g, ind_sub);
                h_sub = sample_vec(h, ind_sub);
                const_pred = (- g_sub.sum() / h_sub.sum());
                best_pred = pred + (const_pred*w);
                pred_sub = sample_vec(best_pred, ind_sub);

                new_initial_score = loss(y_sub, pred_sub, loss_function, w, this);
            }
            

            g = dloss(y, pred, loss_function, this) * w;
            h = ddloss(y, pred, loss_function, this) * w;
            //Rcpp::Rcout << g.array()/h.array() << std::endl;
            


            //and ((g.array())/h.array()).matrix().maxCoeff() > -1e-12

            // Check perfect fit
            if(((g.array())/h.array()).matrix().maxCoeff() < 1e-12 and ((g.array())/h.array()).matrix().maxCoeff() > -1e-12 ){
                // Every perfect step is below tresh
                Rcpp::Rcout  <<
                    std::setprecision(4) <<
                            "max_coeff: " <<((g.array())/h.array()).matrix().maxCoeff()<<
                            "break: "  <<i<<
                            std::endl;
                break;
                
            }
            

            // g, h, X
            //ind_sub = sample_int_rate(n, sample_rate_set);
            g_sub = sample_vec(g, ind_sub);
            h_sub = sample_vec(h, ind_sub);
            X_sub = matrix_subset(X, ind_sub);
            y_sub = sample_vec(y, ind_sub);
            


            
            
            // Train tree
            new_tree->train(g_sub, h_sub, X_sub, cir_sim, greedy_complexities, learning_rate_set);
            //new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
            
            // EXPECTED LOSS
            pred_tmp = new_tree->predict_data(X);
            opt_tmp = new_tree->getTreeOptimism()*learning_rate_set*sample_rate_set;
            total_opt = total_opt + opt_tmp;
            expected_loss = loss_gtb(g, h, pred_tmp) *learning_rate_set*(2 - learning_rate_set) + 
                opt_tmp;
            //gen_loss_change = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
             //   learning_rate_set * new_tree->getTreeOptimism();
            best_pred = learning_rate_set * pred_tmp + best_pred;
            pred_sub = sample_vec(best_pred, ind_sub);

            //directly calculated generalization loss from current ensemble.
            if(gen_loss_type == "R"){
                R_opt += sample_rate_set * ( (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
                learning_rate_set * new_tree->getTreeOptimism() + new_initial_score - loss(y_sub, pred_sub, loss_function, w, this));
                gen_loss = loss(y, best_pred, loss_function, w, this) + R_opt;
                gen_loss_change = gen_loss - best_gen_loss;
            }
            else if(gen_loss_type == "expected"){
                gen_loss_change += expected_loss;
            }
            else if(gen_loss_type == "actual"){
                gen_loss = loss(y, best_pred, loss_function, w, this) + total_opt;
                gen_loss_change = gen_loss - best_gen_loss;
            }
             
            
                
            // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
            if(verbose>0){
                if(i % verbose == 0){
                    Rcpp::Rcout  <<
                        std::setprecision(4) <<
                            "it: " << i << 
                            //"  |  n-leaves: " << new_tree->getNumLeaves() << 
                            "  |  tr loss: " << loss(y, best_pred, loss_function, w, this) <<

                            "  |  gen loss: " << gen_loss_change << 
                            //"  |  highest pred: " << best_pred.maxCoeff() << 
                            //"  |  lowest pred: " << ((-1)*best_pred).maxCoeff() << 
                            // "  |  optimism: " << opt_tmp << 
                            //"  |  reduction: " << gen_loss_change << 
                            "  |  EPS: " << EPS << 
                            "  |  sampSize: " << sample_rate_set << 
                            "  |  best it: " << best_it << 
                            std::endl;
                    
                }
            }

            // if(loss(y, best_pred, loss_function, w, this) != loss(y, best_pred, loss_function, w, this)){

            //     Rcpp::Rcout  <<
            //          std::setprecision(4) <<
            //                 "when nan: "  <<best_pred<<
            //                 std::endl;

            //     break;
            // }
            
            // Stopping criteria 
            // Check for continued learning
            if(force_continued_learning){
                current_tree->next_tree = new_tree;
                current_tree = new_tree;
                pred = best_pred;
                best_opt = total_opt;
            }
            else{
                if(change_criteria == "progressive"){
                    EPS = pow(0.5, round)*last_change_ratio*1.1*best_gen_loss+prog_EPS;

                }
                if(gen_loss_change < EPS){

                    last_change_ratio = gen_loss_change/best_gen_loss;
                    counter = 0;
                    gen_loss_change = 0;
                    improved = true;
                    
                    // sample_rate_set = orig_sample_rate_set; //for shifting rate

                    current_tree->next_tree = new_tree;
                    current_tree = new_tree;
                    current_best_tree = new_tree;
                    pred = best_pred;
                    keep_pred = best_pred;
                    best_gen_loss = gen_loss;
                    best_opt = total_opt;
                    best_R_opt = R_opt;
                    best_it = i;
                    if(include_constant){
                        this->initialPred = this->initialPred + change;
                        change = 0.0;
                        this->initialPred = this->initialPred + const_pred;
                    }

                    if(step_type == "pulse"){
                        if(j>0){
                            j = -1;
                            break;
                        }
                        //Repeat is starting all over again after last iteration if we achieve improvement in the round
                    }
                        
                    
                }
                else{
                    counter += 1;
                    
                    //comment out section below for not including positive reduction trees
                    
                    if(!exclude_bad){
                        current_tree->next_tree = new_tree;
                        current_tree = new_tree;
                        pred = best_pred; 
                        if(include_constant){
                        change += const_pred;
                        }
                    }else{
                        gen_loss_change = 0;
                        total_opt = best_opt;
                    }
                    
                    //best_opt = total_opt;
                   

                    if(counter > MAX_NO_REDUCTION){
                        if(!keep_tail){
                          current_best_tree->next_tree = NULL;
                          current_tree = current_best_tree;
                          pred = keep_pred;
                          best_pred = keep_pred;
                          total_opt = best_opt;
                          R_opt = best_R_opt;
                        }
                        counter = 0;
                        change = 0.0;
                        

                        break;
                        
                    }
                }
            }
            


            i++;
        }

        if(step_type == "repeat"){
                                if(improved){
                                    if(j == (sample_rate.size() - 1)){
                                        round++;
                                        j = -1;
                                        improved = false;
                                    }
                                }
                            }

        // Rcpp::Rcout  <<
        //   std::setprecision(4) <<
        //     "  |  sampSize: " << sample_rate_set << 
        //     "  |  max_NR: " << max_NR << 
        //     std::endl;
    }
}



void ENSEMBLE::train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> g(n), h(n);
    
    // Initial prediction
    g = dloss(y, pred, loss_function, this)*w;
    h = ddloss(y, pred, loss_function, this)*w;
    this->initialPred = - g.sum() / h.sum();
    pred = pred.array() + this->initialPred;
    this->initial_score = loss(y, pred, loss_function, w, this); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = dloss(y, pred, loss_function, this)*w;
    h = ddloss(y, pred, loss_function, this)*w;
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism();
    
    if(verbose>0){
        Rcpp::Rcout  <<
            std::setprecision(4) <<
                "it: " << 1 << 
                    "  |  n-leaves: " << current_tree->getNumLeaves() <<
                        "  |  tr loss: " << loss(y, pred, loss_function, w, this) <<
                            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
                                std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, loss_function, this)*w;
        h = ddloss(y, pred, loss_function, this)*w;
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeOptimism();
        
        // Update preds -- if should not be updated for last iter, it does not matter much computationally
        pred = pred + learning_rate * (new_tree->predict_data(X));
        
        // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
        if(verbose>0){
            if(i % verbose == 0){
                Rcpp::Rcout  <<
                    std::setprecision(4) <<
                        "it: " << i << 
                            "  |  n-leaves: " << current_tree->getNumLeaves() << 
                                "  |  tr loss: " << loss(y, pred, loss_function, w, this) <<
                                    "  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
                                        std::endl;
                
            }
        }
        
        
        if(expected_loss < EPS){ // && NUM_BINTREE_CONSECUTIVE < MAX_NUM_BINTREE_CONSECUTIVE){
            current_tree->next_tree = new_tree;
            current_tree = new_tree;
        }else{
            break;
        }
    }
}

Tvec<double> ENSEMBLE::importance(int ncols)
{
    // Vector with importance
    Tvec<double> importance_vector(ncols);
    importance_vector.setZero();
    
    // Go through each tree to fill importance vector
    GBTREE* current = this->first_tree;
    while(current != NULL)
    {
        current->importance(importance_vector, this->learning_rate);
        current = current->next_tree;
    }
    
    // Scale and return percentwise
    Tvec<double> importance_vec_percent = importance_vector.array()/importance_vector.sum();
    
    return importance_vec_percent;
}

Tvec<double> ENSEMBLE::predict(Tmat<double> &X){
    int n = X.rows();
    int i = 0;
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    GBTREE* current = this->first_tree;
    while(current != NULL){
        // i += 1;
        // Rcpp::Rcout  <<
        //             std::setprecision(4) <<
        //                 "it: " << i << 
        //                 std::endl;
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        current = current->next_tree;
    }
    return pred;
}

Tvec<double> ENSEMBLE::predict2(Tmat<double> &X, int num_trees){
    int n = X.rows();
    int tree_num = 1;
    
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    GBTREE* current = this->first_tree;
    
    
    if(num_trees < 1){
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    
    return pred;
}

Tvec<double> ENSEMBLE::predict3(Tmat<double> &X){
    int n = X.rows();
    int num_trees = this->best_it;
    int tree_num = 1;
    
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    GBTREE* current = this->first_tree;
    
    
    if(num_trees < 1){
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    
    return pred;
}



double ENSEMBLE::estimate_generalization_loss(int num_trees){
  
  int tree_num = 1;
  double total_observed_reduction = 0.0;
  double total_optimism = 0.0;
  double learning_rate = this->learning_rate;
  GBTREE* current = this->first_tree;
  
  if(num_trees<1){
    while(current != NULL){
      total_observed_reduction += current->getTreeScore();
      total_optimism += current->getTreeOptimism();
      current = current->next_tree;
    }
  }else{
    while(current != NULL){
      total_observed_reduction += current->getTreeScore();
      total_optimism += current->getTreeOptimism();
      current = current->next_tree;
      tree_num++;
      if(tree_num > num_trees) break;
    }
  }
  //std::cout<< (this->initial_score) << std::endl;
  return (this->initial_score) + total_observed_reduction * (-2)*learning_rate*(learning_rate/2 - 1) + 
    learning_rate * total_optimism;
  
}


double ENSEMBLE::estimate_generalization_loss_smpl(int num_trees, Tvec<double> &y, Tmat<double> &X){
  

    // Number of trees
    int K = num_trees;
    double gen_loss;
    double total_optimism = 0;
    
    // Prepare prediction vector
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    
    // Unit weights
    Tvec<double> w(n);
    w.setOnes();
    
    GBTREE* current = this->first_tree;
    for(int k=1; k<(K+1); k++)
    {
      // Update predictions with k'th tree
      pred = pred + (this->learning_rate) * (current->predict_data(X));
      
      total_optimism = total_optimism + (this->learning_rate) * (current->sampleRate) * (current->getTreeOptimism());
      // Compute loss
      
      
      // Update to next tree
      current = current->next_tree;
      
      // Check if NULL ptr
      if(current == NULL)
      {
        break;
      }
    }
    gen_loss = loss(y, pred, this->loss_function, w, this) + total_optimism;
    return gen_loss;
}

Tvec<double> ENSEMBLE::get_sample_rates(){
    int num_trees = this->get_num_trees();
    Tvec<double> sample_rates(num_trees);
    GBTREE* current = this->first_tree;
    for(int i=0; i<num_trees; i++){
        sample_rates[i] = current->sampleRate;
        current = current->next_tree;
    }
    return sample_rates;
}

int ENSEMBLE::get_num_trees(){
    int num_trees = 0;
    GBTREE* current = this->first_tree;
    
    while(current != NULL){
        num_trees++;
        current = current->next_tree;
    }
    
    return num_trees;
}

Tvec<double> ENSEMBLE::get_num_leaves(){
    int num_trees = this->get_num_trees();
    Tvec<double> num_leaves(num_trees);
    GBTREE* current = this->first_tree;
    for(int i=0; i<num_trees; i++){
        num_leaves[i] = current->getNumLeaves();
        current = current->next_tree;
    }
    return num_leaves;
}

Tvec<double> ENSEMBLE::convergence(Tvec<double> &y, Tmat<double> &X){
    
    // Number of trees
    int K = this->get_num_trees();
    Tvec<double> loss_val(K+1);
    loss_val.setZero();
    
    // Prepare prediction vector
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    
    // Unit weights
    Tvec<double> w(n);
    w.setOnes();
    
    // After each update (tree), compute loss
    loss_val[0] = loss(y, pred, this->loss_function, w, this);
    
    GBTREE* current = this->first_tree;
    for(int k=1; k<(K+1); k++)
    {
        // Update predictions with k'th tree
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        
        // Compute loss
        loss_val[k] = loss(y, pred, this->loss_function, w, this);
        
        // Update to next tree
        current = current->next_tree;
        
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
    }
    
    return loss_val;
}


// --- GBT_COUNT_AUTO ----
void GBT_COUNT_AUTO::set_param(Rcpp::List par_list){
    this->param = par_list;
    this->learning_rate = par_list["learning_rate"];
    this->extra_param = par_list["extra_param"]; // Starting value
}
Rcpp::List GBT_COUNT_AUTO::get_param(){
    return this->param;
}
GBT_COUNT_AUTO::GBT_COUNT_AUTO(){
    this->count_mod = NULL;
}
ENSEMBLE* GBT_COUNT_AUTO::get_count_mod(){
    return this->count_mod;
}

double GBT_COUNT_AUTO::get_overdispersion(){
    return this->count_mod->get_extra_param();
}

std::string GBT_COUNT_AUTO::get_model_name(){
    std::string count_loss = this->count_mod->get_loss_function();
    if(count_loss == "poisson"){
        return "poisson";
    }else if(count_loss == "negbinom"){
        return "negbinom";
    }else{
        return "unknown";
    }
}

void GBT_COUNT_AUTO::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities)
{
    /*
     * 1. Train Poisson model
     * 2. Learn overdispersion
     * 2.1 If overdispersion large, return Poisson model
     * 3. Train negbinom model
     * 4. Compare relative AIC of models
     * 5. Return and set count model as model with best AIC
     */
    
    
    
    
    // Variables
    double log_factorial;
    double MAX_DISPERSION = 1e9;
    int n =y.size();
    
    // --- 1.0 Poisson ---
    ENSEMBLE* mod_pois = new ENSEMBLE;
    mod_pois->set_param(param["nrounds"], param["learning_rate"], param["extra_param"], "poisson");
    /*
    mod_pois->set_param(
            Rcpp::List::create(
                Named("learning_rate") = param["learning_rate"],
                                              Named("loss_function") = "poisson",
                                              Named("nrounds") = param["nrounds"],
                                                                      Named("extra_param") = param["extra_param"]
            )
    );
    */
    // Training
    Tvec<double> weights = Tvec<double>::Ones(n); // This is unnecessary -- CLEANUP! --> fix ENSEMBLE->train()
    mod_pois->train(y, X, verbose, greedy_complexities, false, weights);

    // ---- 2.0 Learn overdispersion ----
    // Predictions on ynz
    Tvec<double> y_pred_pois = mod_pois->predict(X); // log intensity  
    double dispersion = learn_dispersion(y, y_pred_pois);
    
    // ---- 2.1 Check dispersion -----
    if(dispersion<MAX_DISPERSION)
    {
        // --- 3.1 Train negbinom ----
        ENSEMBLE* mod_nbinom = new ENSEMBLE;
        mod_nbinom->set_param(param["nrounds"], param["learning_rate"], dispersion, "negbinom");
        /*
        mod_nbinom->set_param(
                Rcpp::List::create(
                    Named("learning_rate") = param["learning_rate"],
                                                  Named("loss_function") = "negbinom",
                                                  Named("nrounds") = param["nrounds"],
                                                                          Named("extra_param") = dispersion
                )
        );
         */
        mod_nbinom->train(y, X, verbose, greedy_complexities, false, weights);
        
        // ---- 4. Compare relative AIC of models ----
        Tvec<double> y_pred_nbinom = mod_nbinom->predict(X); // log mean
        dispersion = learn_dispersion(y, y_pred_nbinom, dispersion);
        mod_nbinom->extra_param = dispersion;
        
        // Needs to compare on full likelihood!
        double nll_pois=0.0, nll_nbinom=0.0;
        for(int i=0; i<y.size(); i++)
        {
            // poisson
            log_factorial = 0;
            for(int j=0; j<y[i]; j++){ // also works when y=0-->log_factorial=0, R would have failed...
                log_factorial += log(j+1.0);
            }
            nll_pois -= y[i]*y_pred_pois[i] - exp(y_pred_pois[i]) - log_factorial;
            
            // negative binomial
            nll_nbinom += y[i]*log(dispersion) - y[i]*y_pred_nbinom[i] + 
                (y[i]+dispersion)*log(1.0+exp(y_pred_nbinom[i])/dispersion) - 
                R::lgammafn(y[i]+dispersion) + R::lgammafn(y[i]+1.0) + R::lgammafn(dispersion);
        }
        
        double poisson_aic = nll_pois / y.size();
        double nbinom_aic = (nll_nbinom + 1.0) / y.size();
        
        Rcpp::Rcout << "Relative AIC Poisson: " << poisson_aic << "\n" << 
            "Relative AIC nbinom: " << nbinom_aic << std::endl;
        if(poisson_aic <= nbinom_aic){
            Rcpp::Rcout << "Choosing Poisson model " << std::endl;
            this->count_mod = mod_pois;
        }else{
            Rcpp::Rcout << "Choosing nbinom model " << std::endl;
            this->count_mod = mod_nbinom;
        }
        
    }else{
        // Return with Poisson
        Rcpp::Rcout << "Dispersion too high: Choosing Poisson model " << std::endl;
        this->count_mod = mod_pois;
    }
    
}

Tvec<double> GBT_COUNT_AUTO::predict(Tmat<double> &X)
{
    return this->count_mod->predict(X);
}



// Expose the classes
RCPP_MODULE(aGTBModule) {
    using namespace Rcpp;
    
    class_<ENSEMBLE>("ENSEMBLE")
        .default_constructor("Default constructor")
        .constructor<double>()
        .field("initialPred", &ENSEMBLE::initialPred)
        .method("set_param", &ENSEMBLE::set_param)
        .method("get_nrounds", &ENSEMBLE::get_nrounds)
        .method("get_learning_rate", &ENSEMBLE::get_learning_rate)
        .method("get_extra_param", &ENSEMBLE::get_extra_param)
        .method("get_loss_function", &ENSEMBLE::get_loss_function)
        .method("train", &ENSEMBLE::train)
        .method("sample_train", &ENSEMBLE::sample_train)
        .method("train_from_preds", &ENSEMBLE::train_from_preds)
        .method("predict", &ENSEMBLE::predict)
        .method("predict2", &ENSEMBLE::predict2)
        .method("predict3", &ENSEMBLE::predict3)
        .method("estimate_generalization_loss", &ENSEMBLE::estimate_generalization_loss)
        .method("estimate_generalization_loss_smpl", &ENSEMBLE::estimate_generalization_loss_smpl)
        .method("get_sample_rates", &ENSEMBLE::get_sample_rates)
        .method("get_num_trees", &ENSEMBLE::get_num_trees)
        .method("get_num_leaves", &ENSEMBLE::get_num_leaves)
        .method("save_model", &ENSEMBLE::save_model)
        .method("load_model", &ENSEMBLE::load_model)
        .method("importance", &ENSEMBLE::importance)
        .method("convergence", &ENSEMBLE::convergence)
    ;
    
    class_<GBT_COUNT_AUTO>("GBT_COUNT_AUTO")
        .default_constructor("Default constructor")
        .method("set_param", &GBT_COUNT_AUTO::set_param)
        .method("get_param", &GBT_COUNT_AUTO::get_param)
        .method("train", &GBT_COUNT_AUTO::train)
        .method("predict", &GBT_COUNT_AUTO::predict)
        .method("get_overdispersion", &GBT_COUNT_AUTO::get_overdispersion)
        .method("get_model_name", &GBT_COUNT_AUTO::get_model_name)
    ;
}