'use strict';   // See note about 'use strict'; below

var myApp = angular.module('myAppHome', [
'ngMaterial'
]);

myApp.controller('homeCtrl',['$scope','$http',function ($scope,$http){

  $scope.patient = {
    id:"",
    firstName:"" ,
    lastName:""
  } ;



  $scope.getPatient = function(patient){

    var jsonPatient = JSON.stringify($scope.patient) ;
    $http({
      method: "POST" ,
      url: "/getPatient",
      data:jsonPatient,
      headers:{'Content-Type': 'application/json'}
    }).then(function(response){
      $scope.patient = response.data;
    }
      ,function(error){
        console.log(error) ;
      }
    )
  };



}]);
