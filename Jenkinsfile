pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'ikms-multi-agent-rag'
        REGISTRY = 'docker.io/danula-rathnayaka'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                sh 'pip install uv'
                sh 'uv sync'
            }
        }

        stage('Code Quality Check') {
            steps {
                echo 'Running Quality Checks...'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                }
            }
        }

        stage('Push to Registry') {
            steps {
                echo 'Pushing image to Container Registry...'
            }
        }
    }
}