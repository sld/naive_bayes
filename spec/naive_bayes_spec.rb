require 'rspec'
require_relative "../naive_bayes.rb"


describe "NaiveBayes" do 
  
  before(:each) do
    @training_data = ["Chinese Beijing Chinese",
                                        "Chinese Chinese Shanghai",
                                        "Chinese Macao",
                                        "Tokyo Japan Chinese"]

    @test_data = ["Chinese Chinese Chinese Tokyo Japan"]


    # Classifier sets each word as feature by default
    @nb = NaiveBayes.new
    @training_data[0...3].each{ |data| @nb.train( data, :c ) }
    @nb.train( @training_data[3], :j )      
  end


  it "should correctly calculate conditional probabilites" do
    @nb.cond_prob( "Chinese", :c ).should == 3.0/7
    @nb.cond_prob( "Tokyo", :c ).should == 1.0/14
    @nb.cond_prob( "Tokyo", :j ).should == 2.0/9
    @nb.cond_prob( "Chinese", :j ).should == 2.0/9
  end


  it "should classify to return maximum class probablity :c" do
      @nb.classify( @test_data[0] )[:class].should == :c
  end


  it "should correctly calculate probablity of class" do
    @nb.class_prob(:c).should == 3.0/4
    @nb.class_prob(:j).should == 1.0/4      
  end


  it "should return ~=correctly probabilites for test_data" do
    @nb.document_class_prob( @test_data[0], :c ).should > 0.0003
    @nb.document_class_prob( @test_data[0], :c ).should < 0.0004
    @nb.document_class_prob( @test_data[0], :j ).should > 0.0001
    @nb.document_class_prob( @test_data[0], :j ).should < 0.0002
  end


end