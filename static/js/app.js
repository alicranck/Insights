'use strict'; // See note about 'use strict'; below

var myApp = angular.module('myApp', [
  'ngRoute', 'ngMaterial', 'myAppHome', 'myAppLogin', 'myAppPatientInfo', 'myAppModelInfo', 'ui.router'
]);


myApp.run(function($rootScope) {
  $rootScope.user = {
    id: "",
    name: "",
    position: "",
    loggedIn: true,
  }
})

myApp.controller('rootCtrl', ['$scope', '$rootScope', function($scope, $rootScope) {



}])

myApp.config(function($routeProvider, $locationProvider, $stateProvider) {

    $locationProvider.html5Mode(true)

    var wellcomeState = {
      name: 'wellcome',
      url: '/',
      templateUrl: '/static/partials/index.html'
    }

    var aboutState = {
      name: 'about',
      url: '/about',
      templateUrl: '/static/partials/about.html',
      protected: true,
    }

    var loginState = {
      name: 'login',
      url: '/login',
      templateUrl: '/static/partials/login.html'
    }

    var patientInfoState = {
      name: 'patientInfo',
      url: '/patientInfo',
      templateUrl: '/static/partials/patientInfo/patientInformation.html',
      protected: true,
    }

    var getPatientState = {
      name: 'getPatient',
      url: '/getPatient',
      templateUrl: '/static/partials/patientInfo/getPatient.html',
      protected: true,
    }

    var historyState = {
      name: 'patientInfo.history',
      url: '/history',
      views: {
        'info': {
          templateUrl: '/static/partials/patientInfo/patientHistory.html',
        }
      },
      protected: true,
    }

    var detailsState = {
      name: 'patientInfo.details',
      url: '/details',
      views: {
        'info': {
          templateUrl: '/static/partials/patientInfo/patientDetails.html',
          controller: 'detailsCtrl',
        }
      },
      protected: true,
    }

    var statusState = {
      name: 'patientInfo.status',
      url: '/status',
      views: {
        'info': {
          templateUrl: '/static/partials/patientInfo/status.html',
          controller: 'statusCtrl',
        }
      },
      protected: true,
    }

    var modelInfoState = {
      name: 'modelInfo',
      url: '/modelInfo',
      templateUrl: '/static/partials/modelInfo/landing.html',
      protected: true,
    }

    var modelsState = {
      name: 'modelInfo.models',
      url: '/models',
      views: {
        'modelInfoView': {
          templateUrl: '/static/partials/modelInfo/models.html',
          controller: 'modelsCtrl',
        }
      },
      protected: true,
    }

    var clustersState = {
      name: 'modelInfo.clusters',
      url: '/clusters',
      views: {
        'modelInfoView': {
          templateUrl: '/static/partials/modelInfo/clusters.html',
          controller: 'clustersCtrl',
        }
      },
      protected: true,
    }

    var statsState = {
      name: 'modelInfo.stats',
      url: '/stats',
      views: {
        'modelInfoView': {
          templateUrl: '/static/partials/modelInfo/stats.html',
          controller: 'statsCtrl',
        }
      },
      protected: true,
    }



    $stateProvider.state(wellcomeState);
    $stateProvider.state(aboutState);
    $stateProvider.state(loginState);
    $stateProvider.state(patientInfoState);
    $stateProvider.state(historyState);
    $stateProvider.state(detailsState);
    $stateProvider.state(statusState);
    $stateProvider.state(getPatientState);
    $stateProvider.state(modelInfoState);
    $stateProvider.state(modelsState);
    $stateProvider.state(clustersState);
    $stateProvider.state(statsState);



  })
  .run(function($rootScope, $transitions) {
    $transitions.onBefore({}, function(transition) {
      // check if the state should be protected
      if (transition.to().protected && !$rootScope.user.loggedIn) {
        // redirect to the 'login' state
        return transition.router.stateService.target('login');
      }
    });
  })
  .run(function($rootScope, $transitions) {
    $transitions.onBefore({
      to: 'login',
    }, function(transition) {
      // check if already logged in
      if ($rootScope.user.loggedIn) {
        // redirect to the landing state
        return transition.router.stateService.target('getPatient');
      }
    });
  })

$(document).click(function(e) {
  if (true) {
    $('.collapse').collapse('hide');
  }
});
