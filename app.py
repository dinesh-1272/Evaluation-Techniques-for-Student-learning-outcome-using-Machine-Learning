from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    data1=request.form['age']
    data2=request.form['gender']
    def gender(gender_input):
        if gender_input=='Male':
            return 1
        else:
            return 0
    data_gen=gender(data2)
    data3=request.form['class_size']
    data4=request.form['family_size']
    data5=request.form['father_education']
    def fatedu(fatedu_input):
        if fatedu_input=='Diploma':
            return 0
        elif fatedu_input=='Graduate':
            return 1
        elif fatedu_input=='High School':
            return 2
        else:
            return 3
    data_fe=fatedu(data5)
    data6=request.form['mother_education']
    def motedu(motedu_input):
        if motedu_input=='Diploma':
            return 0
        elif motedu_input=='Graduate':
            return 1
        elif motedu_input=='High School':
            return 2
        else:
            return 3
    data_me=motedu(data6)
    data7=request.form['father_occupation']
    def fatocc(fatocc_input):
        if fatocc_input=='Business':
            return 0
        elif fatocc_input=='Government':
            return 1
        elif fatocc_input=='Private':
            return 2
        else:
            return 3
    data_fo=fatocc(data7)
    data8=request.form['mother_occupation']
    def motocc(motocc_input):
        if motocc_input=='Business':
            return 0
        elif motocc_input=='Housewife':
            return 1
        elif motocc_input=='Government':
            return 2
        elif motocc_input=='Private':
            return 3
        else:
            return 4
    data_mo=motocc(data8)
    data9=request.form['health_issues']
    def hi(hi_input):
        if hi_input=='Yes':
            return 1
        else:
            return 0
    data_hi=hi(data9)
    data10=request.form['parent_marital_status']
    def pms(pms_input):
        if pms_input=='Married':
            return 0
        elif pms_input=='Single parent':
            return 1
        else:
            return 2
    data_pms=pms(data10)
    data11=request.form['practice_sport']
    def ps(ps_input):
        if ps_input=='Never':
            return 0
        elif ps_input=='Regular':
            return 1
        else:
            return 2
    data_ps=ps(data11)
    data12=request.form['attendance']
    def att(att_input):
        if att_input=='above 70%':
            return 0
        elif att_input=='above 40%':
            return 1
        else:
            return 2
    data_att=att(data12)
    data13=request.form['homework_completion']
    def hc(hc_input):
        if hc_input=='Always':
            return 0
        elif hc_input=='Never':
            return 1
        else:
            return 2
    data_hc=hc(data13)
    data14=request.form['academic_score']
    def aca(aca_input):
        if aca_input=='Excellent(above 90%)':
            return 0
        elif aca_input=='Good(above 60%)':
            return 1
        elif aca_input=='Low(above 40%)':
            return 2
        else:
            return 3
    data_aca=aca(data14)
    data15=request.form['attentivity_in_class']
    def aic(aic_input):
        if aic_input=='Attentive':
            return 0
        elif aic_input=='Sometimes attentive':
            return 1
        else:
            return 2
    data_aic=aic(data15)
    data16=request.form['behavioral_patterns']
    def bp(bp_input):
        if bp_input=='Aggressive':
            return 0
        elif bp_input=='Calm':
            return 1
        elif bp_input=='Focused':
            return 2
        elif bp_input=='Hyper active':
            return 3
        else:
            return 4
    data_bp=bp(data16)
    data17=request.form['self_esteem']
    def se(se_input):
        if se_input=='Confident':
            return 0
        elif se_input=='Low':
            return 1
        else:
            return 2
    data_se=se(data17)
    data18=request.form['socially_skills']
    def ss(ss_input):
        if ss_input=='Active':
            return 0
        elif ss_input=='Inactive':
            return 1
        else:
            return 2
    data_ss=ss(data18)
    data19=request.form['teacher_interaction']
    def ti(ti_input):
        if ti_input=='Good':
            return 0
        elif ti_input=='Moderate':
            return 1
        else:
            return 2
    data_ti=ti(data19)
    data20=request.form['cognitive_development']
    def cd(cd_input):
        if cd_input=='Excellent':
            return 0
        elif cd_input=='Good':
            return 1
        else:
            return 2
    data_cd=cd(data20)
    data21=request.form['technology_influence']
    def tec(tec_input):
        if tec_input=='Excellent':
            return 0
        elif tec_input=='Good':
            return 1
        else:
            return 2
    data_tec=tec(data21)
    data22=request.form['social_media_influence']
    def smi(smi_input):
        if smi_input=='Inactive':
            return 0
        elif smi_input=='Moderately active':
            return 1
        else:
            return 2
    data_smi=smi(data22)
    data23=request.form['extra_curricular_involvement']
    def eci(eci_input):
        if eci_input=='Never':
            return 0
        elif eci_input=='Regular':
            return 1
        else:
            return 2
    data_eci=eci(data11)
    arr=np.array([[data1,data_gen,data3,data4,data_fe,data_me,data_fo,data_mo,data_hi,data_pms,data_ps,data_att,data_hc,data_aca,data_aic,data_bp,data_se,data_ss,data_ti,data_cd,data_tec,data_smi,data_eci]])
    pred=model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == '__main__':
    app.run(debug=True)