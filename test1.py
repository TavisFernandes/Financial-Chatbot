from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
import socket
from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django import forms
from django.core.mail import EmailMessage
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import random
import requests
from fuzzywuzzy import process

# Define SignupForm and LoginForm directly here
class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    class Meta:
        model = User
        fields = ['username', 'email', 'password']

class LoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

# Define Userd model directly here
from django.db import models

class Userd(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    account_bal = models.DecimalField(max_digits=10, decimal_places=2)
    credit_score = models.IntegerField()
    base_limit = models.DecimalField(max_digits=10, decimal_places=2)

# Define account activation token generator
from django.utils.crypto import salted_hmac

class TokenGenerator:
    def make_token(self, user):
        return salted_hmac(
            key_salt="django.contrib.auth.tokens.PasswordResetTokenGenerator",
            value=str(user.pk) + str(user.is_active),
        ).hexdigest()

    def check_token(self, user, token):
        return self.make_token(user) == token

account_activation_token = TokenGenerator()

credit_card_thresh = 300
credit_score = 0
bal = 0
email = None
base_limit = 0

class Links:
    checkout_message = "You can checkout further information in the following link: "
    personal_loan_link = "127.0.0.1:8000/chatbot/EducationalLoans"
    error_msg = "Sorry, we couldn't find any response!!"

def get_info_from_db_dict(parameter, db_info):
    choices = list(db_info)
    db_col = process.extractOne(parameter, choices)[0]
    return db_info[db_col]

def get_general_info_from_db(parameter):
    db_info = {'contact': "Bhushan: 8898546254"}
    return get_info_from_db_dict(parameter, db_info)

def get_user_specific_info(parameter):
    global base_limit, bal, email, credit_score
    db_info = {'credit score': 350, 'balance': 20000 , 'base limit': 5000, 'uname': 'Bhushan Borole'}
    return get_info_from_db_dict(parameter, db_info)

class IntentMapping:
    @staticmethod
    def card_charge(length, entity, value):
        if length > 0 and entity == "card_type":
            if value == "debit card":
                return "For debit card first year is free and from second year onward we charge 300tk per year"
            elif value == "credit card":
                return "For credit cards we charge 500% per year"
            else:
                return "I am not sure about that, sorry!"
        else:
            return "For credit cards we charge 500% per year. And for debit cards first year is free and from second year we charge 300tk per year"

    @staticmethod
    def show_balance():
        return "Your current account balance is {}".format(get_user_specific_info('balance'))

    @staticmethod
    def summary():
        bal = get_user_specific_info('balance')
        base_limit = get_user_specific_info('base limit')
        return "Account type: Checking Account\nCurrent balance: {}.\nAvailable to withdraw: {}.".format(bal, bal - base_limit)

    @staticmethod
    def greet():
        try:
            uname = get_user_specific_info('user name')
            if uname is not None:
                return "Hello, {}. How may I help you??".format(uname)
            else:
                return "Hello user, How may I help you??"
        except:
            return "Hello user, How may I help you??"

get_random_response = lambda intent: "Default response for this intent."

isClosingCard = False
chequeBookAddress = 0

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        start = None
        try:
            start = request.POST['text']   
            print(start)

            response2 = requests.get("http://localhost:5000/parse", params={"q": request.POST["text"]})
            print(response2)

            response2 = response2.json()
            intent = response2["intent"]["name"]
            entities = response2["entities"]
            length = len(entities)
            
            if length > 0:
                entity = entities[0]["entity"]
                value = entities[0]["value"]
            else:
                entity, value = None, None

            if intent == "event-request":
                response_text = get_event(entities["day"], entities["time"], entities["place"])
            elif intent == "greet":
                response_text = IntentMapping.greet()
            elif intent == "card_charge":
                response_text = IntentMapping.card_charge(length, entity, value)
            elif intent == "get_cheque_book":
                response_text = IntentMapping.get_cheque_book()
            elif intent == "loan_car":
                response_text = IntentMapping.loan_personal()
            elif intent == "loan_home":
                response_text = "We provide home loan of minimum 20 lacs tk and maximum 1.2 crore tk"
            elif intent == "loan_max":
                response_text = IntentMapping.loan_max(length, entity, value)
            elif intent == "loan_min":
                response_text = IntentMapping.loan_min(length, value, entity)
            elif intent == "loan_details":
                response_text = IntentMapping.loan_details()
            elif intent == "show_balance":
                response_text = IntentMapping.show_balance()
            elif intent == "summary":
                response_text = IntentMapping.summary()
            elif intent == 'lost_card':
                response_text = IntentMapping.lost_card()
            elif intent == "loan_personal":
                response_text = IntentMapping.loan_personal()
            else:
                response_text = "Sorry, I don't have a response for this intent."

            global chequeBookAddress
            print(chequeBookAddress, "chequeBookAddress")
            if chequeBookAddress == 1:
                _response = JsonResponse({"status": "success", "response": "Enter your address: "})
                chequeBookAddress = chequeBookAddress + (1 << 1)
                return _response

            elif chequeBookAddress == 3:
                if len(request.POST['text']) > 5:
                    chequeBookAddress = 0
                    return JsonResponse({
                        "status": "success",
                        "response": "Cheque book will be delivered to your address: \n{}." 
                                   "\nNominal fees will be deducted from your bank account.".format(request.POST['text'])
                    })
                else:
                    return JsonResponse({"status": "success", "response": "Please enter a valid address."})

            else:
                if not response_text:
                    response_text = ask_q(request.POST['text'])
                return JsonResponse({"status": "success", "response": response_text})

        except Exception as e:
            print(e)
            from traceback import print_exc
            print_exc()
            response2 = requests.get("http://localhost:5000/parse", params={"q": request.POST["text"]}).json()
            return JsonResponse({"status": "success", "response": str(response2["intent"])})

    return render(request, "index.html", {"start": start})

chequeBookAddress = 0

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        start = request.POST.get('text', '')
        response_text = IntentMapping.greet()
        return JsonResponse({"status": "success", "response": response_text})
    return render(request, "index.html", {"start": None})

@csrf_exempt
def post_register(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            current_site = request.get_host()
            message = render_to_string('account_activation_email.html', {
                'user': user,
                'domain': current_site,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
            mail_subject = 'Activate your account.'
            email = EmailMessage(mail_subject, message, to=[form.cleaned_data['email']])
            email.send()
            return render(request, 'register.html', {'form': form, 'message': 'Please confirm your email address to complete the registration'})
    else:
        form = SignupForm()
    return render(request, 'register.html', {'form': form})
