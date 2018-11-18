from django.shortcuts import render, redirect
from django.views import View
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import login as auth_login
from django.contrib.auth import authenticate
from django.contrib.auth import logout as auth_logout
from django.core.exceptions import ObjectDoesNotExist

from polls import views
from polls.models import AuthUser

class signup(View) :
    def get(self, request, *args, **kwargs) :
        return render(request, 'user/signup.html')

    def post(self, request, *args, **kwargs) :
        print("SignUp Post:\n{}\n".format(request.POST))
        
        condition = False

        response = HttpResponse("Failed Sign Up")
        createID = request.POST['uname']
        createPW = request.POST['psw']
        createPW_repeat = request.POST['psw-repeat']
        createE = request.POST['email']

        createGender = request.POST['gender']
        createYear = request.POST['year']
        createMonth = request.POST['month']
        createDay = request.POST['day']
        
        try :
            User.objects.get(username = createID)
        except ObjectDoesNotExist :
            condition = True
        
        if createPW != createPW_repeat :
            return render(request, 'user/login.html', {'not_equal' : True})

        if not condition :
            return render(request, 'user/login.html', {'overlap' : True})

        else :
            '''
            user = AuthUser(username = createID,
                            email = createE,
                            password = createPW,
                            gender = createGender,
                            birthday = createYear + '-' + createMonth + '-' + createDay)
            user.save()
            '''

            user = User.objects.create_user(username = createID, 
                                            email = createE, 
                                            password = createPW)
            user.save()
            
            user = AuthUser.objects.get(username = createID)            
            user.birthday = createYear + '-' + createMonth + '-' + createDay
            user.gender = createGender
            user.save()

            return render(request, 'user/login.html')

class login(View) :
    def get(self, request, *args, **kwargs) :
        return render(request, 'user/login.html')

    def post(self, request, *args, **kwargs) :
        condition = False

        userID = request.POST['uname']
        userPW = request.POST['psw']

        user = authenticate(request, username = userID, password = userPW)
    
        if user is not None : # 접속한 유저의 ID를 같이 넘겨줌
            auth_login(request, user)
            return redirect(views.index)

        else :
            return render(request, 'user/login.html', {'diff' : True})
    
def logout(request) :
    auth_logout(request)
    return redirect('user/login')
