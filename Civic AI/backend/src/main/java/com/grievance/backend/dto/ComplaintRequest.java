package com.grievance.backend.dto;

public class ComplaintRequest {
    private String name;
    private String email;
    private String phone;
    private String district;
    private String address;
    private String complaintTitle;
    private String complaintDescription;

    public ComplaintRequest(){
    }
    public String getName(){
        return name;
    }
    public void setName(String name){
        this.name=name;
    }
    public String getEmail(){
        return email;
    }
    public void setEmail(String email){
        this.email=email;
    }
    public String getPhone(){
        return phone;
    }
    public void setPhone(String phone){
        this.phone=phone;
    }
    public String getDistrict(){
        return district;
    }
    public void setDistrict(String district){
        this.district=district;
    }
    public String getAddress(){
        return address;
    }
    public void setAddress(String address){
        this.address=address;
    }
    public String getComplaintTitle(){
        return complaintTitle;
    }
    public void setComplaintTitle(String complaintTitle){
        this.complaintTitle=complaintTitle;
    }
    public String getComplaintDescription(){
        return complaintDescription;
    }
    public void setComplaintDescription(String complaintDescription){
        this.complaintDescription=complaintDescription;
    }


}
