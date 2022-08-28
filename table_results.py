import numpy as np

def printlog(s, log_name):
    print(s, file=open("results/"+log_name+".txt", "a"))

def loadAlgData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    length = len(lines)
    
    model = [] 
    alg1_aver = [] 
    alg1_std = []
    alg1_t = [] 
    sample_num = []
    
    for line in lines:
        if "models" in line:
            s = line.split(" ")[1]
            s = s.split("/")
            length = len(s)
            s = s[length-1]
            # s = s.replace("\n", "")
            model.append(s)
        elif "mean" in line:
            aver = line.split(" ")[0]
            aver = float(aver.split("=")[1])
            std = line.split(" ")[1]
            std = float(std.split("=")[1])
            alg1_aver.append(aver)
            alg1_std.append(std)
        elif "average" in line:
            time = float(line.split(" ")[1])
            alg1_t.append(time)
        elif "images" in line:
            num = int(line.split(" ")[6])
            sample_num.append(num)
        
    return model, alg1_aver, alg1_std, alg1_t, sample_num    

def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    length = len(lines)
    
    dataset = []
    model = [] 
    NW_aver = [] 
    DC_aver =[] 
    VN_aver = [] 
    RV_aver = [] 
    NW_std = []
    DC_std = [] 
    VN_std = [] 
    RV_std = [] 
    NW_t = [] 
    DC_t = [] 
    VN_t = []
    RV_t = []
    
    for line in lines:
        if "======" in line:
            continue
        elif "model" in line:
            s = line.split(" ")[3]
            s = s.split("/")
            length = len(s)
            s = s[length-1]
            s = s.replace("\n", "")
            dataset.append(s.split("_")[0])
            model.append(s)
        elif "NeWise" in line:
            aver = line.split(",")[2]
            aver = float(aver.split("=")[1])
            std = line.split(",")[3]
            std = float(std.split("=")[1])
            time = line.split(",")[4]
            time = float(time.split("=")[1])
            NW_aver.append(aver)
            NW_std.append(std)
            NW_t.append(time)
        elif "DeepCert" in line:
            aver = line.split(",")[2]
            aver = float(aver.split("=")[1])
            std = line.split(",")[3]
            std = float(std.split("=")[1])
            time = line.split(",")[4]
            time = float(time.split("=")[1])
            DC_aver.append(aver)
            DC_std.append(std)
            DC_t.append(time)
        elif "VeriNet" in line:
            aver = line.split(",")[2]
            aver = float(aver.split("=")[1])
            std = line.split(",")[3]
            std = float(std.split("=")[1])
            time = line.split(",")[4]
            time = float(time.split("=")[1])
            VN_aver.append(aver)
            VN_std.append(std)
            VN_t.append(time)
        elif "RobustVerifier" in line:
            aver = line.split(",")[2]
            aver = float(aver.split("=")[1])
            std = line.split(",")[3]
            std = float(std.split("=")[1])
            time = line.split(",")[4]
            time = float(time.split("=")[1])
            RV_aver.append(aver)
            RV_std.append(std)
            RV_t.append(time)
        
    return dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t

if __name__ == '__main__':
    
    log_name = "table_results"
    
    # Table 1
    file_path = "logs/table_one.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 1"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {}".format("Model", "DeepCert", "VeriNet", "Rob.Ver."), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:.4f}".format(model[i], DC_aver[i], VN_aver[i], RV_aver[i]), log_name)
        if((i+1 < len(dataset)) and (dataset[i] == 'mnist') and (dataset[i+1] == 'fashion')):
            printlog("-"*250, log_name)
        elif ((i+1 < len(dataset)) and (dataset[i] == 'fashion') and (dataset[i+1] == 'cifar10')):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 2
    file_path = "logs/table_two.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 2"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {:>10}".format("Model", "NW_aver", "DC_aver", "Impr.(%)", "VN_aver", "Impr.(%)", "RV_aver","Impr.(%)", "NW_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)", "Time(s)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        aver = (NW_t[i] + DC_t[i] + VN_t[i] + RV_t[i]) / 4
        ma = max(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        mi = min(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        plus = max(ma-aver, aver-mi)
        
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:>6.2f} {:>4.2f}".format(model[i], NW_aver[i], DC_aver[i], (NW_aver[i]-DC_aver[i])/DC_aver[i]*100, VN_aver[i], (NW_aver[i]-VN_aver[i])/VN_aver[i]*100, RV_aver[i], (NW_aver[i]-RV_aver[i])/RV_aver[i]*100, NW_std[i], DC_std[i], (NW_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (NW_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (NW_std[i]-RV_std[i])/RV_std[i]*100, aver, plus), log_name)
        if((i+1 < len(dataset)) and (dataset[i] == 'mnist') and (dataset[i+1] == 'fashion')):
            printlog("-"*250, log_name)
        elif ((i+1 < len(dataset)) and (dataset[i] == 'fashion') and (dataset[i+1] == 'cifar10')):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 3
    file_path = "logs/table_three.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 3"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format("Model", "NW_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f}".format(model[i], NW_std[i], DC_std[i], (NW_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (NW_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (NW_std[i]-RV_std[i])/RV_std[i]*100), log_name)
        if((i+1 < len(dataset)) and (dataset[i] == 'mnist') and (dataset[i+1] == 'fashion')):
            printlog("-"*250, log_name)
        elif ((i+1 < len(dataset)) and (dataset[i] == 'fashion') and (dataset[i+1] == 'cifar10')):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 4
    file_path = "logs/table_four.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    file_path = "logs/alg1_table_4.txt"
    model, alg1_aver, alg1_std, alg1_t, sample_num = loadAlgData(file_path)
    
    table_name = "Table 4"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {}".format("Model", "Alg.1_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f}".format(model[i], alg1_std[i], DC_std[i], (alg1_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (alg1_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (alg1_std[i]-RV_std[i])/RV_std[i]*100), log_name)
        if((i+1 < len(dataset)) and ("cnn" in model[i]) and ("fnn" in model[i+1])):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 5
    file_path = "logs/table_five.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 5"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {:>10}".format("Model", "NW_aver", "DC_aver", "Impr.(%)", "VN_aver", "Impr.(%)", "RV_aver","Impr.(%)", "NW_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)", "Time(s)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        aver = (NW_t[i] + DC_t[i] + VN_t[i] + RV_t[i]) / 4
        ma = max(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        mi = min(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        plus = max(ma-aver, aver-mi)
        
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:>6.2f} {:>4.2f}".format(model[i], NW_aver[i], DC_aver[i], (NW_aver[i]-DC_aver[i])/DC_aver[i]*100, VN_aver[i], (NW_aver[i]-VN_aver[i])/VN_aver[i]*100, RV_aver[i], (NW_aver[i]-RV_aver[i])/RV_aver[i]*100, NW_std[i], DC_std[i], (NW_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (NW_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (NW_std[i]-RV_std[i])/RV_std[i]*100, aver, plus), log_name)
        if((i+1 < len(dataset)) and ("cnn" in model[i]) and ("fnn" in model[i+1])):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 6
    file_path = "logs/table_six.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 6"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {:>10}".format("Model", "NW_aver", "DC_aver", "Impr.(%)", "VN_aver", "Impr.(%)", "RV_aver","Impr.(%)", "NW_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)", "Time(s)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        aver = (NW_t[i] + DC_t[i] + VN_t[i] + RV_t[i]) / 4
        ma = max(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        mi = min(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        plus = max(ma-aver, aver-mi)
        
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:>6.2f} {:>4.2f}".format(model[i], NW_aver[i], DC_aver[i], (NW_aver[i]-DC_aver[i])/DC_aver[i]*100, VN_aver[i], (NW_aver[i]-VN_aver[i])/VN_aver[i]*100, RV_aver[i], (NW_aver[i]-RV_aver[i])/RV_aver[i]*100, NW_std[i], DC_std[i], (NW_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (NW_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (NW_std[i]-RV_std[i])/RV_std[i]*100, aver, plus), log_name)
        if((i+1 < len(dataset)) and (dataset[i] == 'mnist') and (dataset[i+1] == 'fashion')):
            printlog("-"*250, log_name)
        elif ((i+1 < len(dataset)) and (dataset[i] == 'fashion') and (dataset[i+1] == 'cifar10')):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 7
    file_path = "logs/table_seven.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 7"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {:>10}".format("Model", "NW_aver", "DC_aver", "Impr.(%)", "VN_aver", "Impr.(%)", "RV_aver","Impr.(%)", "NW_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)", "Time(s)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        aver = (NW_t[i] + DC_t[i] + VN_t[i] + RV_t[i]) / 4
        ma = max(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        mi = min(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        plus = max(ma-aver, aver-mi)
        
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:>6.2f} {:>4.2f}".format(model[i], NW_aver[i], DC_aver[i], (NW_aver[i]-DC_aver[i])/DC_aver[i]*100, VN_aver[i], (NW_aver[i]-VN_aver[i])/VN_aver[i]*100, RV_aver[i], (NW_aver[i]-RV_aver[i])/RV_aver[i]*100, NW_std[i], DC_std[i], (NW_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (NW_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (NW_std[i]-RV_std[i])/RV_std[i]*100, aver, plus), log_name)
        if((i+1 < len(dataset)) and (dataset[i] == 'mnist') and (dataset[i+1] == 'fashion')):
            printlog("-"*250, log_name)
        elif ((i+1 < len(dataset)) and (dataset[i] == 'fashion') and (dataset[i+1] == 'cifar10')):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 8
    file_path = "logs/table_eight.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    table_name = "Table 8"
    printlog("="*250, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {:>10}".format("Model", "NW_aver", "DC_aver", "Impr.(%)", "VN_aver", "Impr.(%)", "RV_aver","Impr.(%)", "NW_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)", "Time(s)"), log_name)
    printlog("-"*250, log_name)
    
    for i in range(len(dataset)):
        aver = (NW_t[i] + DC_t[i] + VN_t[i] + RV_t[i]) / 4
        ma = max(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        mi = min(NW_t[i], DC_t[i], VN_t[i], RV_t[i])
        plus = max(ma-aver, aver-mi)
        
        printlog("{:65} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:.4f} \t {:>6.2f} \t {:>6.2f} {:>4.2f}".format(model[i], NW_aver[i], DC_aver[i], (NW_aver[i]-DC_aver[i])/DC_aver[i]*100, VN_aver[i], (NW_aver[i]-VN_aver[i])/VN_aver[i]*100, RV_aver[i], (NW_aver[i]-RV_aver[i])/RV_aver[i]*100, NW_std[i], DC_std[i], (NW_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (NW_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (NW_std[i]-RV_std[i])/RV_std[i]*100, aver, plus), log_name)
        if((i+1 < len(dataset)) and (dataset[i] == 'mnist') and (dataset[i+1] == 'fashion')):
            printlog("-"*250, log_name)
        elif ((i+1 < len(dataset)) and (dataset[i] == 'fashion') and (dataset[i+1] == 'cifar10')):
            printlog("-"*250, log_name)
    printlog("="*250, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
    
    # Table 9
    file_path = "logs/table_nine.txt"
    dataset, model, NW_aver, DC_aver, VN_aver, RV_aver, NW_std, DC_std, VN_std, RV_std, NW_t, DC_t, VN_t, RV_t = loadData(file_path)
    
    file_path = "logs/alg1_table_9.txt"
    model, alg1_aver, alg1_std, alg1_t, sample_num = loadAlgData(file_path)
    
    table_name = "Table 9"
    printlog("="*325, log_name)
    printlog("{}: \n".format(table_name), log_name)
    printlog("{:65} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10} \t {:>10}".format("Model", "Alg.1_aver", "DC_aver", "Impr.(%)", "VN_aver", "Impr.(%)", "RV_aver","Impr.(%)", "Alg.1_std", "DC_std", "Impr.(%)", "VN_std", "Impr.(%)", "RV_std", "Impr.(%)", "Alg.1 Time(s)", "Others Time(s)"), log_name)
    printlog("-"*325, log_name)
    
    for i in range(len(dataset)):
        aver = (DC_t[i] + VN_t[i] + RV_t[i]) / 3
        ma = max(DC_t[i], VN_t[i], RV_t[i])
        mi = min(DC_t[i], VN_t[i], RV_t[i])
        plus = max(ma-aver, aver-mi)
        
        printlog("{:65} \t {:>10.4f} \t {:>10.4f} \t {:>10.2f} \t {:>10.4f} \t {:>10.2f} \t {:>10.4f} \t {:>10.2f} \t {:>10.4f} \t {:>10.4f} \t {:>10.2f} \t {:>10.4f} \t {:>10.2f} \t {:>10.4f} \t{:>10.2f} \t {:>10.2f} \t {:>10.2f} {:.2f}".format(model[i], alg1_aver[i], DC_aver[i], (alg1_aver[i]-DC_aver[i])/DC_aver[i]*100, VN_aver[i], (alg1_aver[i]-VN_aver[i])/VN_aver[i]*100, RV_aver[i], (alg1_aver[i]-RV_aver[i])/RV_aver[i]*100, alg1_std[i], DC_std[i], (alg1_std[i]-DC_std[i])/DC_std[i]*100, VN_std[i], (alg1_std[i]-VN_std[i])/VN_std[i]*100, RV_std[i], (alg1_std[i]-RV_std[i])/RV_std[i]*100, alg1_t[i], aver, plus), log_name)
        if((i+1 < len(dataset)) and ("cnn" in model[i]) and ("fnn" in model[i+1])):
            printlog("-"*325, log_name)
    printlog("="*325, log_name)
    printlog("\n \n \n", log_name)
            
    print(table_name, "generated!!!")
